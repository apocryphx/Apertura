#!/usr/bin/env ruby
# Wires the AperturaKit product framework (v2 — the user-created AperturaKit target with
# its synchronized AperturaKit/ folder is the product; aptransformer reverts to the pure
# engine framework). Idempotent, in-place (never recreates targets — the colocate phase
# and mlx paths on existing targets must survive).
#
#   AperturaKit target   <- facade + tokenizer/template (auto via synchronized folder)
#                           + engine ES*.mm (explicit refs) + libmlx + ObjCTokenizer.framework
#                           (built by the WORKSPACE: code/Apertura.xcworkspace) + metallib phase
#   aptransformer target <- engine only, as before phase 2 (revert product rename, drop OCT)
#   AperturaResearch CLI <- compiles facade directly from AperturaKit/ (include/ shim symlink)
#
# Public headers live in the AperturaKit folder's exception set (edited textually after —
# see wire_aperturakit_headers.py; the xcodeproj gem's coverage of synchronized-group
# exception sets is not trusted for writes).
require 'xcodeproj'

ROOT = File.expand_path(File.join(__dir__, '..'))
PROJ = File.join(ROOT, 'Apertura.xcodeproj')
OCT  = File.expand_path(File.join(ROOT, '..', 'ObjCTokenizer', 'ObjCTokenizer'))

# Xcode 26 writes multi-line shellScript values as ARRAYS of lines; the xcodeproj gem
# only accepts the string form. Normalize before opening (idempotent; Xcode reads both).
pbx = File.join(PROJ, 'project.pbxproj')
src = File.read(pbx)
normalized = src.gsub(/shellScript = \(\n(.*?)\n\t*\);/m) do
  lines = $1.scan(/"((?:[^"\\]|\\.)*)"\s*,\s*\n?/).flatten
  "shellScript = \"#{lines.join('\n')}\";"
end
File.write(pbx, normalized) if normalized != src

project = Xcodeproj::Project.open(PROJ)
fw   = project.targets.find { |t| t.name == 'aptransformer' }    or abort 'no aptransformer target'
kit  = project.targets.find { |t| t.name == 'AperturaKit' }      or abort 'no AperturaKit target'
tool = project.targets.find { |t| t.name == 'AperturaResearch' } or abort 'no AperturaResearch target'

AP_IMPLS = Dir[File.join(ROOT, 'AperturaKit', 'AP*.{m,mm}')].sort +
           %w[ESTokenizer.mm ESChatTemplate.mm].map { |f| File.join(ROOT, 'AperturaKit', f) }
ES_MM    = Dir[File.join(ROOT, 'aptransformer', 'ES*.mm')].sort

def group(project, name)
  project.main_group[name] || project.main_group.new_group(name)
end

def ensure_source(project, target, grp, path)
  return if target.source_build_phase.files.any? { |bf|
    bf.file_ref && bf.file_ref.real_path.to_s == path }
  ref = project.files.find { |r| r.real_path.to_s == path } || grp.new_reference(path)
  target.source_build_phase.add_file_reference(ref, true)
end

# ---- prune refs whose recorded path no longer exists (the AP/tokenizer moves) ----
project.files.select { |r|
  p = r.path.to_s
  (p.start_with?('aptransformer/AP') || p =~ %r{aptransformer/ES(Tokenizer|ChatTemplate)}) &&
    !File.exist?(r.real_path.to_s)
}.each do |r|
  project.targets.each do |t|
    (t.build_phases.flat_map(&:files) rescue []).select { |bf| bf.file_ref == r }
                                                 .each(&:remove_from_project)
  end
  r.remove_from_project
end

# ---- aptransformer: back to the pure engine framework ----
fw.source_build_phase.files.select { |bf|
  p = bf.file_ref && bf.file_ref.real_path.to_s
  p && (p.include?('/ObjCTokenizer/')) }.each(&:remove_from_project)
fw.build_configurations.each do |c|
  bs = c.build_settings
  bs['PRODUCT_NAME'] = 'aptransformer'
  bs.delete('PRODUCT_MODULE_NAME')
  bs.delete('DEFINES_MODULE')
  bs['HEADER_SEARCH_PATHS'] = Array(bs['HEADER_SEARCH_PATHS']).reject { |p| p.include?('ObjCTokenizer') }
end

# ---- AperturaKit: engine sources + build settings + metallib phase + OCT link ----
core_grp = group(project, 'AperturaCore-src')
ES_MM.each { |p| ensure_source(project, kit, core_grp, p) }

header_paths = ['$(inherited)', '/opt/homebrew/include', '$(SRCROOT)/../mlx',
                '$(SRCROOT)/aptransformer', '$(SRCROOT)/AperturaKit',
                File.dirname(OCT), OCT, File.join(OCT, 'Internal')]
ldflags = ['$(inherited)', '-lmlx', '-licucore',
           '-framework', 'Foundation', '-framework', 'Metal',
           '-framework', 'Accelerate', '-framework', 'QuartzCore',
           '-framework', 'MetalPerformanceShaders']
kit.build_configurations.each do |c|
  bs = c.build_settings
  bs['HEADER_SEARCH_PATHS']           = header_paths
  bs['LIBRARY_SEARCH_PATHS']          = ['$(inherited)', '$(SRCROOT)/../mlx/build', '/opt/homebrew/lib']
  bs['OTHER_LDFLAGS']                 = ldflags
  bs['CLANG_CXX_LANGUAGE_STANDARD']   = 'gnu++20'
  bs['CLANG_CXX_LIBRARY']             = 'libc++'
  bs['GCC_WARN_INHIBIT_ALL_WARNINGS'] = 'YES'
  bs['MACOSX_DEPLOYMENT_TARGET']      = '14.0'
  bs['MLX_METALLIB'] = '$(SRCROOT)/../mlx/build/mlx/backend/metal/kernels/mlx.metallib'
end

phase = kit.shell_script_build_phases.find { |p| p.name == 'Colocate mlx.metallib' } ||
        kit.new_shell_script_build_phase('Colocate mlx.metallib')
# The metallib ships in the framework's RESOURCES (codesign forbids non-code files next
# to the binary in Versions/A). APModel points MLX at it via metal::set_metallib_path().
phase.shell_script = "# libmlx.a bakes in a default metallib path from its own build tree; the framework\n" \
                     "# ships the metallib as a resource and APModel sets the override path at load.\n" \
                     "cp -f \"${MLX_METALLIB}\" \"${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/\"\n"
# Declared inputs/outputs keep the user-script SANDBOX (new-target default) happy and
# make the copy incremental-build-aware.
phase.input_paths  = ['$(MLX_METALLIB)']
phase.output_paths = ['$(TARGET_BUILD_DIR)/$(UNLOCALIZED_RESOURCES_FOLDER_PATH)/mlx.metallib']

oct_ref = project.files.find { |r| File.basename(r.path.to_s) == 'ObjCTokenizer.framework' }
abort 'no ObjCTokenizer.framework ref (expected from the app target)' unless oct_ref
unless kit.frameworks_build_phase.files.any? { |bf| bf.file_ref == oct_ref }
  kit.frameworks_build_phase.add_file_reference(oct_ref, true)
end

# ---- AperturaKitTests: standalone framework tests (the template created it app-hosted;
# a host app drags the whole legacy embed chain into `xcodebuild test`) ----
if (kt = project.targets.find { |t| t.name == 'AperturaKitTests' })
  kt.build_configurations.each do |c|
    c.build_settings.delete('TEST_HOST')
    c.build_settings.delete('BUNDLE_LOADER')
  end
end

# ---- Apertura app: consume AperturaKit (drop the legacy aptransformer embed; keep
# ObjCTokenizer.framework — AperturaKit links it as a dylib dependency) ----
if (app = project.targets.find { |t| t.name == 'Apertura' })
  [app.frameworks_build_phase, *app.copy_files_build_phases].each do |ph|
    ph.files.select { |bf|
      bf.file_ref && File.basename(bf.file_ref.path.to_s) == 'aptransformer.framework'
    }.each(&:remove_from_project)
  end
  kit_ref = kit.product_reference
  unless app.frameworks_build_phase.files.any? { |bf| bf.file_ref == kit_ref }
    app.frameworks_build_phase.add_file_reference(kit_ref, true)
  end
  embed = app.copy_files_build_phases.find { |p| (p.name || '') =~ /Embed/ }
  if embed && embed.files.none? { |bf| bf.file_ref == kit_ref }
    bf = embed.add_file_reference(kit_ref, true)
    bf.settings = { 'ATTRIBUTES' => %w[CodeSignOnCopy RemoveHeadersOnCopy] }
  end
  app.add_dependency(kit)
  # Research posture: the app loads multi-GB model bundles + persona files from arbitrary
  # local paths (the Models volume), which App Sandbox forbids. Deliberately disabled —
  # matches the CLI. Revisit (NSOpenPanel + security-scoped bookmarks) for distribution.
  app.build_configurations.each { |c| c.build_settings['ENABLE_APP_SANDBOX'] = 'NO' }
end

# ---- CLI: facade + tokenizer/template from their new home ----
cli_grp = group(project, 'AperturaKit-cli')
AP_IMPLS.each { |p| ensure_source(project, tool, cli_grp, p) }
tool.build_configurations.each do |c|
  hp = Array(c.build_settings['HEADER_SEARCH_PATHS'] || ['$(inherited)'])
  ['$(SRCROOT)/AperturaResearch/include', '$(SRCROOT)/AperturaKit'].each { |p| hp << p unless hp.include?(p) }
  c.build_settings['HEADER_SEARCH_PATHS'] = hp
end

project.save

# The gem drops Xcode 26's symbolic `dstSubfolder` from PBXCopyFilesBuildPhase on save;
# restore the numeric spec (10 = Frameworks) wherever it is now missing.
patched = File.read(pbx).gsub(/(isa = PBXCopyFilesBuildPhase;\n\t*dstPath = "";\n)(?!\t*dstSubfolderSpec)/) do
  "#{$1}\t\t\tdstSubfolderSpec = 10;\n"
end
File.write(pbx, patched)

puts "kit: +#{ES_MM.size} engine sources, OCT framework linked, metallib phase ensured"
puts "aptransformer reverted to engine framework; CLI repointed to AperturaKit/"
