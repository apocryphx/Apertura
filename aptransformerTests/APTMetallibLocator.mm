//
//  APTMetallibLocator — point MLX at the metallib shipped in the test bundle.
//
//  The test bundle links libmlx.a directly, whose default metallib path bakes in the
//  absolute location of the machine-local mlx build tree. The "Colocate mlx.metallib"
//  phase copies the metallib into the bundle's Resources (codesign forbids non-code
//  files in Contents/MacOS, and MLX's automatic lookup never checks an .xctest's
//  Resources), so set the override path before any kernel loads — the same override
//  APModel uses for the AperturaKit framework.
//

#import <Foundation/Foundation.h>
#include "mlx/backend/metal/metal.h"

@interface APTMetallibLocator : NSObject
@end

@implementation APTMetallibLocator

+ (void)load {
    NSString * lib = [[NSBundle bundleForClass:self] pathForResource:@"mlx" ofType:@"metallib"];
    if (lib) mlx::core::metal::set_metallib_path(std::string(lib.UTF8String));
}

@end
