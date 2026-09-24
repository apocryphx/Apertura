#!/usr/bin/env ruby
# Reconfigures aptransformerTests into a self-contained logic-test bundle:
#  - clears TEST_HOST / app dependency (no GUI app launch, no 58 GB model)
#  - compiles the ES* sources directly + links libmlx
#  - the synchronized aptransformerTests/ folder provides the test .mm
require 'xcodeproj'

ROOT = File.expand_path(File.join(__dir__, '..'))
PROJ = File.join(ROOT, 'Apertura.xcodeproj')
project = Xcodeproj::Project.open(PROJ)

t = project.targets.find { |x| x.name == 'aptransformerTests' } or abort 'aptransformerTests not found'

# Drop app/framework coupling -> standalone logic test.
t.dependencies.dup.each(&:remove_from_project)
t.frameworks_build_phase.files.dup.each do |f|
  f.remove_from_project if f.display_name.to_s.include?('aptransformer.framework')
end

# Compile the ES sources into the test target (regular group, explicit refs).
grp = project.main_group['AperturaCore-src'] || project.main_group.new_group('AperturaCore-src')
existing = t.source_build_phase.files_references.map { |r| r.real_path.to_s rescue nil }.compact
Dir[File.join(ROOT, 'aptransformer', 'ES*.mm')].sort.each do |p|
  next if existing.include?(p)
  ref = project.files.find { |f| (f.real_path.to_s rescue nil) == p } || grp.new_reference(p)
  t.source_build_phase.add_file_reference(ref, true)
end

t.build_configurations.each do |c|
  bs = c.build_settings
  bs.delete('TEST_HOST'); bs['TEST_HOST'] = ''
  bs['BUNDLE_LOADER'] = ''
  bs['HEADER_SEARCH_PATHS']         = ['$(inherited)', '$(MLX_ROOT)', '$(SRCROOT)/aptransformer']
  bs['LIBRARY_SEARCH_PATHS']        = ['$(inherited)', '$(MLX_BUILD)']
  bs['OTHER_LDFLAGS']               = ['$(inherited)', '-lmlx',
                                       '-framework', 'Metal', '-framework', 'Accelerate',
                                       '-framework', 'QuartzCore', '-framework', 'MetalPerformanceShaders']
  bs['CLANG_CXX_LANGUAGE_STANDARD'] = 'gnu++20'
  bs['CLANG_CXX_LIBRARY']           = 'libc++'
  bs['CLANG_ENABLE_OBJC_ARC']       = 'YES'
  bs['GCC_WARN_INHIBIT_ALL_WARNINGS'] = 'YES'
end

# The test bundle links libmlx.a directly, whose default metallib path bakes in the
# machine-local mlx build tree. Ship the metallib in the bundle's RESOURCES (codesign
# forbids non-code files in Contents/MacOS); APTMetallibLocator.mm sets the override
# path at load, mirroring APModel's handling for the framework.
phase = t.shell_script_build_phases.find { |p| p.name == 'Colocate mlx.metallib' } ||
        t.new_shell_script_build_phase('Colocate mlx.metallib')
phase.shell_script = "# libmlx.a bakes in a default metallib path from its own build tree; the test bundle\n" \
                     "# ships the metallib in Resources (codesign forbids non-code files in Contents/MacOS)\n" \
                     "# and APTMetallibLocator sets the override path at load.\n" \
                     "cp -f \"${MLX_METALLIB}\" \"${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/\"\n"
phase.input_paths  = ['$(MLX_METALLIB)']
phase.output_paths = ['$(TARGET_BUILD_DIR)/$(UNLOCALIZED_RESOURCES_FOLDER_PATH)/mlx.metallib']

project.save

# Shared scheme that runs the tests.
scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(t, true)
test_action = scheme.test_action
tref = Xcodeproj::XCScheme::TestAction::TestableReference.new(t)
test_action.add_testable(tref)
scheme.save_as(PROJ, 'aptransformerTests', true)

puts "aptransformerTests -> logic test; ES sources compiled in; libmlx linked."
