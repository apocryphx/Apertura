//
//  PersonaViewController.h
//  Apertura
//
//  The custom-GPT editor pane: name + one body text view (sections separated by the
//  \n\n---\n\n convention), with the version chain browsable through a back/forward
//  stepper. Saving snapshots the old state first (CDPersona's archive-first rule);
//  history rows are read-only.

#import <Cocoa/Cocoa.h>
#import "CDPersona.h"

NS_ASSUME_NONNULL_BEGIN

@interface PersonaViewController : NSViewController

/// Show a persona (a HEAD row). Pass nil to blank the pane.
- (void)showPersona:(nullable CDPersona *)persona;

@end

NS_ASSUME_NONNULL_END
