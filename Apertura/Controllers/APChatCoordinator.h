//
//  APChatCoordinator.h
//  Apertura
//
//  The engine/session/persistence half of what used to be ViewController (M3 of the
//  management plan): model load, persona priming with KV-snapshot restore, streaming
//  turns, attachment staging and budgeting, Core Data archiving, resume (replay and
//  checkpoint fast paths), the headless-quit checkpoint, and the persona tools. Owns no
//  views — everything the user sees goes out through APChatCoordinatorDelegate, and the
//  chat view forwards user intent back in. Behavior is a straight port of the v1 flows;
//  the delegate calls are one-to-one with the old rendering calls.

#import <Cocoa/Cocoa.h>
#import <AperturaKit/AperturaKit.h>

@class APChatCoordinator, CDChatSession, CDPersona;

NS_ASSUME_NONNULL_BEGIN

@protocol APChatCoordinatorDelegate <NSObject>
- (void)coordinatorClearTranscript:(APChatCoordinator *)coordinator;
- (void)coordinator:(APChatCoordinator *)coordinator appendSpeakerHeader:(NSString *)name;
- (void)coordinator:(APChatCoordinator *)coordinator appendSpeaker:(NSString *)name text:(NSString *)text;
- (void)coordinator:(APChatCoordinator *)coordinator appendStreamedText:(NSString *)text;
- (void)coordinator:(APChatCoordinator *)coordinator appendNote:(NSString *)text;
- (void)coordinator:(APChatCoordinator *)coordinator appendDelta:(APResponseDelta *)delta;
- (void)coordinator:(APChatCoordinator *)coordinator didChangeStatus:(NSString *)status busy:(BOOL)busy;
/// Read composeEnabled / sessionControlsEnabled / stopEnabled and update the controls.
- (void)coordinatorDidChangeControls:(APChatCoordinator *)coordinator;
/// Read stagedAttachmentSummary / stagedAttachmentCount and update the bar.
- (void)coordinatorDidChangeAttachments:(APChatCoordinator *)coordinator;
- (void)coordinator:(APChatCoordinator *)coordinator didChangeWindowTitle:(NSString *)title;
- (void)coordinatorRequestInputFocus:(APChatCoordinator *)coordinator;
/// Resume adopts the row's reasoning flag; a declined API key falls back to local —
/// keep the checkbox/popup in sync.
- (void)coordinator:(APChatCoordinator *)coordinator didAdoptReasoning:(BOOL)reasoning;
- (void)coordinator:(APChatCoordinator *)coordinator didAdoptBackendGoogle:(BOOL)google;
@end

@interface APChatCoordinator : NSObject

@property (weak, nullable) id<APChatCoordinatorDelegate> delegate;

#pragma mark - State the views read

@property (readonly) BOOL composeEnabled;          // input field + attach button
@property (readonly) BOOL sessionControlsEnabled;  // reasoning toggle, backend popup, resume
@property (readonly) BOOL stopEnabled;
@property (readonly, nullable) NSString * stagedAttachmentSummary;   // nil when none staged
@property (readonly) NSUInteger stagedAttachmentCount;
@property (readonly) BOOL googleBackendSelected;
@property (readonly) BOOL reasoningEnabled;        // the persisted default the session uses

#pragma mark - Lifecycle

/// Kick off the current backend: load the model if needed, then prime (or resume the
/// device checkpoint on first call). Call once when the UI appears; again on backend or
/// reasoning changes.
- (void)startForCurrentBackend;

#pragma mark - User intent (forwarded by the chat view)

- (void)sendText:(NSString *)text;
- (void)stopGeneration;
- (void)setReasoningEnabled:(BOOL)reasoning;       // restarts the conversation
- (void)setGoogleBackendSelected:(BOOL)google;     // restarts the conversation
/// Stage files to ride the next user turn. Returns how many were accepted; every
/// rejection explains itself in the transcript.
- (NSUInteger)stageAttachmentsAtURLs:(NSArray<NSURL *> *)urls;
- (void)clearAttachments;

#pragma mark - Sessions (sidebar)

/// The custom GPT new conversations prime from. Defaults to the first stored persona;
/// when none exist, the legacy persona FILE is imported as one on first use.
@property (nonatomic, nullable) CDPersona * activePersona;

/// Start a fresh conversation, optionally with a specific GPT (nil = keep the active
/// one). The current conversation's checkpoint is saved in passing when it has one.
- (void)startNewChatWithPersona:(nullable CDPersona *)persona;

/// Re-open a stored conversation (persona + turns re-primed; the row's own checkpoint
/// is the fast path). The current conversation checkpoint-saves in passing. No-op while
/// a turn is streaming.
- (void)resumeChatSession:(CDChatSession *)row;

#pragma mark - Termination (AppDelegate)

/// Headless-quit checkpoint: starts the async KV save and returns YES (call
/// `completion` on the main queue when done), or returns NO when there is nothing to
/// checkpoint.
- (BOOL)beginTerminationCheckpointWithCompletion:(void (^)(void))completion;

@end

NS_ASSUME_NONNULL_END
