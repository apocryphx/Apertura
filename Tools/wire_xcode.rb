#!/usr/bin/env ruby
# Wires MLX + Apertura sources into the existing Apertura.xcodeproj without regenerating it.
#  - aptransformer framework: MLX build settings + all ES* sources/headers.
#  - AperturaResearch CLI target: ES* sources + main.mm, links -lmlx directly.
require 'xcodeproj'

PROJ = File.expand_path(File.join(__dir__, '..', 'Apertura.xcodeproj'))
ROOT = File.expand_path(File.join(__dir__, '..'))
project = Xcodeproj::Project.open(PROJ)

ES_DIR = 'aptransformer'
ES_BASENAMES = %w[
  ESModelConfig ESWeightLoader ESRMSNorm ESRotaryEmbedding ESMLPBlock ESKVCache
  ESAttention ESDecoderLayer ESGemma4TextModel ESGemma4TextForCausalLM
  ESSampler ESGenerationLoop ESConformance
]
ES_MM      = ES_BASENAMES.map { |b| "#{ES_DIR}/#{b}.mm" }
ES_HEADERS = (ES_BASENAMES + %w[ESOps]).map { |b| "#{ES_DIR}/#{b}.h" }

def mlx_settings(c, cxx_only: false)
  bs = c.build_settings
  bs['HEADER_SEARCH_PATHS']      = ['$(inherited)', '/opt/homebrew/include']
  bs['LIBRARY_SEARCH_PATHS']     = ['$(inherited)', '/opt/homebrew/lib']
  bs['OTHER_LDFLAGS']            = ['$(inherited)', '-lmlx']
  bs['LD_RUNPATH_SEARCH_PATHS']  = ['$(inherited)', '/opt/homebrew/lib']
  bs['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++17'
  bs['CLANG_CXX_LIBRARY']        = 'libc++'
  bs['GCC_C_LANGUAGE_STANDARD']  = 'gnu17'
end

# ---- find-or-create a file reference under a group, by repo-relative path ----
def file_ref(project, group, rel_path)
  abs = File.expand_path(File.join(File.dirname(project.path), '..', rel_path)) rescue nil
  existing = project.files.find { |f| f.real_path.to_s.end_with?(rel_path) }
  return existing if existing
  group.new_reference(File.expand_path(File.join(ROOT, rel_path)))
end

# Groups
es_group = project.main_group[ES_DIR] || project.main_group.new_group(ES_DIR, ES_DIR)
research_group = project.main_group['AperturaResearch'] || project.main_group.new_group('AperturaResearch', 'AperturaResearch')

# ---- aptransformer framework target ----
fw = project.targets.find { |t| t.name == 'aptransformer' }
raise 'aptransformer target not found' unless fw
fw.build_configurations.each do |c|
  mlx_settings(c)
  c.build_settings['ENABLE_MODULE_VERIFIER'] = 'NO'
  c.build_settings['DEFINES_MODULE'] = 'YES'
end

# add ES sources + headers to framework
ES_MM.each do |rel|
  ref = file_ref(project, es_group, rel)
  fw.source_build_phase.add_file_reference(ref, true) unless fw.source_build_phase.files_references.include?(ref)
end
ES_HEADERS.each do |rel|
  file_ref(project, es_group, rel) # ensure referenced in project tree
end

# ---- AperturaResearch CLI target ----
research = project.targets.find { |t| t.name == 'AperturaResearch' }
unless research
  research = project.new_target(:command_line_tool, 'AperturaResearch', :osx, '13.0')
end
research.build_configurations.each do |c|
  mlx_settings(c)
  c.build_settings['PRODUCT_NAME'] = 'AperturaResearch'
  c.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  c.build_settings['DEVELOPMENT_TEAM'] = '2PYWYF3C55'
  c.build_settings['CLANG_ENABLE_OBJC_ARC'] = 'YES'
end

# research compiles all ES sources + main.mm directly (no framework runtime dependency)
main_ref = file_ref(project, research_group, 'AperturaResearch/main.mm')
(ES_MM + ['AperturaResearch/main.mm']).each do |rel|
  grp = rel.start_with?('AperturaResearch') ? research_group : es_group
  ref = file_ref(project, grp, rel)
  unless research.source_build_phase.files_references.include?(ref)
    research.source_build_phase.add_file_reference(ref, true)
  end
end

project.save

# ---- shared scheme for AperturaResearch ----
scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(research)
scheme.set_launch_target(research)
scheme.save_as(PROJ, 'AperturaResearch', true)

puts "wired: aptransformer (#{ES_MM.size} sources) + AperturaResearch CLI (#{ES_MM.size + 1} sources)"
