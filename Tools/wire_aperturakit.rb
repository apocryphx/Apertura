#!/usr/bin/env ruby
# Wires the AperturaKit facade into the project IN PLACE (no target recreation, so the
# later-added settings — metallib colocation, mlx search paths — survive):
#  - framework target 'aptransformer' -> product/module AperturaKit; adds the AP facade
#    sources, the promoted ESTokenizer/ESChatTemplate, and the ObjCTokenizer sources;
#    marks AP*.h + AperturaKit.h PUBLIC (engine ES*.h stay project-visibility).
#  - CLI target 'AperturaResearch' -> re-points the moved tokenizer/template refs and
#    adds the AP facade sources (the CLI compiles sources directly; --facade-verify
#    gates the facade against the reference session path).
# Idempotent: safe to re-run.
require 'xcodeproj'

ROOT = File.expand_path(File.join(__dir__, '..'))
PROJ = File.join(ROOT, 'Apertura.xcodeproj')
OCT  = File.expand_path(File.join(ROOT, '..', 'ObjCTokenizer', 'ObjCTokenizer'))

project = Xcodeproj::Project.open(PROJ)
fw   = project.targets.find { |t| t.name == 'aptransformer' }    or abort 'no aptransformer target'
tool = project.targets.find { |t| t.name == 'AperturaResearch' } or abort 'no AperturaResearch target'

AP_IMPLS   = Dir[File.join(ROOT, 'aptransformer', 'AP*.{m,mm}')].sort
AP_HEADERS = (Dir[File.join(ROOT, 'aptransformer', 'AP*.h')] - [File.join(ROOT, 'aptransformer', 'APInternal.h')]).sort
UMBRELLA   = File.join(ROOT, 'aptransformer', 'AperturaKit.h')
PROMOTED   = %w[ESTokenizer.mm ESChatTemplate.mm].map { |f| File.join(ROOT, 'aptransformer', f) }
OCT_M      = (Dir[File.join(OCT, '*.m')] + Dir[File.join(OCT, 'Internal', '*.m')] +
              Dir[File.join(OCT, 'Vendor', 'yyjson', '*.c')]).sort

def group(project, name)
  project.main_group[name] || project.main_group.new_group(name)
end

# Remove stale build files / refs matching a basename predicate (dangling paths etc.).
def prune(project, basenames)
  project.targets.each do |t|
    (t.build_phases.flat_map(&:files) rescue []).select { |bf|
      bf.file_ref && basenames.include?(File.basename(bf.file_ref.path.to_s))
    }.each(&:remove_from_project)
  end
  project.files.select { |r| basenames.include?(File.basename(r.path.to_s)) &&
                             !File.exist?(r.real_path.to_s) }.each(&:remove_from_project)
end

def ensure_source(project, target, grp, path)
  existing = target.source_build_phase.files.find { |bf|
    bf.file_ref && bf.file_ref.real_path.to_s == path }
  return if existing
  ref = project.files.find { |r| r.real_path.to_s == path } || grp.new_reference(path)
  target.source_build_phase.add_file_reference(ref, true)
end

def ensure_public_header(project, target, grp, path)
  ref = project.files.find { |r| r.real_path.to_s == path } || grp.new_reference(path)
  bf = target.headers_build_phase.files.find { |f| f.file_ref && f.file_ref.real_path.to_s == path }
  bf ||= target.headers_build_phase.add_file_reference(ref, true)
  bf.settings = { 'ATTRIBUTES' => ['Public'] }
end

# ---- prune: the deleted umbrella + the moved tokenizer/template old refs ----
prune(project, ['aptransformer.h'])
# moved files: drop refs whose recorded path no longer exists (old AperturaResearch/ location)
%w[ESTokenizer.h ESTokenizer.mm ESChatTemplate.h ESChatTemplate.mm].each do |base|
  project.files.select { |r| File.basename(r.path.to_s) == base && !File.exist?(r.real_path.to_s) }
         .each do |r|
    project.targets.each do |t|
      (t.build_phases.flat_map(&:files) rescue []).select { |bf| bf.file_ref == r }
                                                   .each(&:remove_from_project)
    end
    r.remove_from_project
  end
end

kit_grp = group(project, 'AperturaKit')
oct_grp = group(project, 'ObjCTokenizer-src')

# ---- framework target ----
# The aptransformer/ folder is a SYNCHRONIZED group on this target: everything in it
# (ES engine, promoted tokenizer/template, AP facade) is auto-membered — explicit source
# entries would double-compile, so remove any that point into the folder. Public-header
# marking happens in the folder's exception set (edited separately). Only the
# out-of-folder ObjCTokenizer sources need explicit membership.
fw.source_build_phase.files.select { |bf|
  bf.file_ref && bf.file_ref.real_path.to_s.include?('/aptransformer/')
}.each(&:remove_from_project)
OCT_M.each { |p| ensure_source(project, fw, oct_grp, p) }

fw.build_configurations.each do |c|
  bs = c.build_settings
  bs['PRODUCT_NAME']         = 'AperturaKit'
  bs['PRODUCT_MODULE_NAME']  = 'AperturaKit'
  bs['DEFINES_MODULE']       = 'YES'
  hp = Array(bs['HEADER_SEARCH_PATHS'] || ['$(inherited)'])
  [File.dirname(OCT), OCT, File.join(OCT, 'Internal')].each { |p| hp << p unless hp.include?(p) }
  bs['HEADER_SEARCH_PATHS'] = hp
end

# ---- CLI target: moved paths + facade sources ----
# The CLI compiles the facade directly (no framework link); the include/ shim directory
# (AperturaKit -> ../../aptransformer symlink) lets the public headers' canonical
# <AperturaKit/...> imports resolve in this context too.
(AP_IMPLS + PROMOTED).each { |p| ensure_source(project, tool, kit_grp, p) }
tool.build_configurations.each do |c|
  hp = Array(c.build_settings['HEADER_SEARCH_PATHS'] || ['$(inherited)'])
  shim = '$(SRCROOT)/AperturaResearch/include'
  hp << shim unless hp.include?(shim)
  c.build_settings['HEADER_SEARCH_PATHS'] = hp
end

project.save
puts "framework: +#{AP_IMPLS.size} AP impls, +#{PROMOTED.size} promoted, +#{OCT_M.size} OCT, #{AP_HEADERS.size + 1} public headers"
puts "cli      : facade + promoted sources ensured"
