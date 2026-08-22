#!/usr/bin/env ruby
# Creates the apertura-mcp command-line target: the MCP stdio server over AperturaKit.
# Compiles the AperturaMCP/ sources plus the app-side files it shares with the app
# (Core Data stack + entities, model registry) and the compiled data model; links the
# AperturaKit framework and copies it (with ObjCTokenizer) beside the binary. Build the
# WORKSPACE — the frameworks come from there. Idempotent.
require 'xcodeproj'

ROOT = File.expand_path(File.join(__dir__, '..'))
PROJ = File.join(ROOT, 'Apertura.xcodeproj')

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

project.targets.select { |t| t.name == 'apertura-mcp' }.each(&:remove_from_project)

tool = project.new_target(:command_line_tool, 'apertura-mcp', :osx, '14.0')

def add_sources(project, target, group_name, paths)
  grp = project.main_group[group_name] || project.main_group.new_group(group_name)
  paths.each do |p|
    ref = grp.new_reference(p)
    target.source_build_phase.add_file_reference(ref, true)
  end
end

mcp_sources = Dir[File.join(ROOT, 'AperturaMCP', '*.m')].sort
shared = [File.join(ROOT, 'Apertura', 'Core Data', 'APPersistence.m'),
          File.join(ROOT, 'Apertura', 'Core Data', 'CDChatSession.m'),
          File.join(ROOT, 'Apertura', 'Core Data', 'CDPersona.m'),
          File.join(ROOT, 'Apertura', 'Model Registry', 'APModelRegistry.m')]

add_sources(project, tool, 'AperturaMCP', mcp_sources)
add_sources(project, tool, 'AperturaMCP-shared', shared)

# The data model compiles per-target (momc + property codegen) — it belongs in the
# SOURCES phase; the momd lands in the products directory for a CLI target, where
# APPersistence's beside-the-executable fallback finds it.
model_ref = (project.main_group['AperturaMCP-shared'] || project.main_group.new_group('AperturaMCP-shared'))
                .new_reference(File.join(ROOT, 'Apertura', 'Apertura.xcdatamodeld'))
tool.source_build_phase.add_file_reference(model_ref, true)

# Link + embed AperturaKit (which itself needs ObjCTokenizer beside it).
kit = project.targets.find { |t| t.name == 'AperturaKit' }
tool.add_dependency(kit)
kit_product = kit.product_reference
tool.frameworks_build_phase.add_file_reference(kit_product, true)

embed = tool.new_copy_files_build_phase('Embed Frameworks')
embed.symbol_dst_subfolder_spec = :products_directory
embed.dst_path = ''
# AperturaKit.framework is already in Products next to the binary from its own build;
# copying is still declared so a standalone product folder stays self-contained.
embed.add_file_reference(kit_product, true)

tool.build_configurations.each do |c|
  bs = c.build_settings
  bs['HEADER_SEARCH_PATHS'] = ['$(inherited)',
                               '$(SRCROOT)/Apertura/Core Data',
                               '$(SRCROOT)/Apertura/Model Registry',
                               '$(SRCROOT)/AperturaMCP']
  bs['FRAMEWORK_SEARCH_PATHS'] = ['$(inherited)', '$(BUILT_PRODUCTS_DIR)']
  bs['LD_RUNPATH_SEARCH_PATHS'] = ['$(inherited)', '@executable_path', '@loader_path']
  bs['CLANG_ENABLE_OBJC_ARC'] = 'YES'
  bs['PRODUCT_NAME'] = 'apertura-mcp'
  bs['MACOSX_DEPLOYMENT_TARGET'] = '14.0'
  bs['CODE_SIGN_STYLE'] = 'Automatic'
  bs['DEVELOPMENT_TEAM'] = '2PYWYF3C55'
end

project.save

scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(tool)
scheme.set_launch_target(tool)
scheme.save_as(PROJ, 'apertura-mcp', true)

puts "apertura-mcp target: #{mcp_sources.size} MCP + #{shared.size} shared sources"
