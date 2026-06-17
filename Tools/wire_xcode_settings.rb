#!/usr/bin/env ruby
# Settings-only: add MLX build settings to the aptransformer framework target.
# (Sources are auto-included via Xcode 26 file-system-synchronized folders.)
require 'xcodeproj'

PROJ = File.expand_path(File.join(__dir__, '..', 'Apertura.xcodeproj'))
project = Xcodeproj::Project.open(PROJ)

fw = project.targets.find { |t| t.name == 'aptransformer' } or abort 'aptransformer target not found'
fw.build_configurations.each do |c|
  bs = c.build_settings
  bs['HEADER_SEARCH_PATHS']         = ['$(inherited)', '/opt/homebrew/include']
  bs['LIBRARY_SEARCH_PATHS']        = ['$(inherited)', '/opt/homebrew/lib']
  bs['OTHER_LDFLAGS']               = ['$(inherited)', '-lmlx']
  bs['LD_RUNPATH_SEARCH_PATHS']     = ['$(inherited)', '@loader_path/Frameworks', '/opt/homebrew/lib']
  bs['CLANG_CXX_LANGUAGE_STANDARD'] = 'gnu++20'
  bs['CLANG_CXX_LIBRARY']           = 'libc++'
  bs['ENABLE_MODULE_VERIFIER']      = 'NO'
  bs['DEFINES_MODULE']              = 'YES'
  bs['GCC_WARN_INHIBIT_ALL_WARNINGS'] = 'YES' # MLX headers emit many C++20-extension notes
end

project.save
puts "ok: MLX settings applied to aptransformer framework"
