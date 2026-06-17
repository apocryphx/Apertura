#!/usr/bin/env ruby
# Creates the AperturaResearch command-line target in the existing project.
# Compiles ES* (framework sources), the driver, ESTokenizer, and the ObjCTokenizer
# sources directly; links libmlx + libicucore + system frameworks. Xcode compiles each
# file by extension (.m -> ObjC, .mm -> ObjC++), so mixed-language sources are fine here.
require 'xcodeproj'

ROOT = File.expand_path(File.join(__dir__, '..'))
PROJ = File.join(ROOT, 'Apertura.xcodeproj')
OCT  = File.expand_path('~/Documents/GitHub/ObjCTokenizer/ObjCTokenizer')

project = Xcodeproj::Project.open(PROJ)

# Remove a prior AperturaResearch target if present (idempotent re-runs).
project.targets.select { |t| t.name == 'AperturaResearch' }.each(&:remove_from_project)

tool = project.new_target(:command_line_tool, 'AperturaResearch', :osx, '13.0')

# ---- source file references (regular groups; avoids synchronized-group APIs) ----
def add_sources(project, target, group_name, paths)
  grp = project.main_group[group_name] || project.main_group.new_group(group_name)
  paths.each do |p|
    ref = grp.new_reference(p)
    target.source_build_phase.add_file_reference(ref, true)
  end
end

es_mm = Dir[File.join(ROOT, 'aptransformer', 'ES*.mm')].sort
oct_m = (Dir[File.join(OCT, '*.m')] + Dir[File.join(OCT, 'Internal', '*.m')]).sort
driver = [File.join(ROOT, 'AperturaResearch', 'main.mm'),
          File.join(ROOT, 'AperturaResearch', 'ESTokenizer.mm')]

add_sources(project, tool, 'AperturaCore-src', es_mm)
add_sources(project, tool, 'AperturaResearch', driver)
add_sources(project, tool, 'ObjCTokenizer-src', oct_m)

# ---- build settings ----
header_paths = ['$(inherited)', '/opt/homebrew/include',
                '$(SRCROOT)/aptransformer', '$(SRCROOT)/AperturaResearch',
                File.dirname(OCT), OCT, File.join(OCT, 'Internal')]
ldflags = ['$(inherited)', '-lmlx', '-licucore',
           '-framework', 'Foundation', '-framework', 'Metal',
           '-framework', 'Accelerate', '-framework', 'QuartzCore',
           '-framework', 'MetalPerformanceShaders']

tool.build_configurations.each do |c|
  bs = c.build_settings
  bs['HEADER_SEARCH_PATHS']         = header_paths
  bs['LIBRARY_SEARCH_PATHS']        = ['$(inherited)', '/opt/homebrew/lib']
  bs['OTHER_LDFLAGS']               = ldflags
  bs['CLANG_CXX_LANGUAGE_STANDARD'] = 'gnu++20'
  bs['CLANG_CXX_LIBRARY']           = 'libc++'
  bs['CLANG_ENABLE_OBJC_ARC']       = 'YES'
  bs['GCC_WARN_INHIBIT_ALL_WARNINGS'] = 'YES'
  bs['PRODUCT_NAME']                = 'AperturaResearch'
  bs['MACOSX_DEPLOYMENT_TARGET']    = '13.0'
  bs['CODE_SIGN_STYLE']             = 'Automatic'
  bs['DEVELOPMENT_TEAM']            = '2PYWYF3C55'
end

project.save

scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(tool)
scheme.set_launch_target(tool)
scheme.save_as(PROJ, 'AperturaResearch', true)

puts "AperturaResearch target: #{es_mm.size} ES + #{driver.size} driver + #{oct_m.size} OCT sources"
