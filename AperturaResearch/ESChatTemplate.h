#pragma once
//  ESChatTemplate — Gemma-4 chat formatting + response parsing for the driver.
//
//  Gemma-4 replaced Gemma-2/3's `<start_of_turn>`/`<end_of_turn>` scheme with a paired
//  bracket-token grammar, where every marker is a single reserved low-id token in the shared
//  Gemma-4 tokenizer vocab:
//
//     <|turn>105  <turn|>106        a turn:   <|turn>{role}\n ... <turn|>\n
//     <|think|>98                   "thinking enabled" flag (top of the system turn)
//     <|channel>100 <channel|>101   a channel:  <|channel>thought\n ... <channel|>
//     <|tool>46 <tool|>47           tool DECLARATION block (schema)
//     <|tool_call>48 <tool_call|>49 model emits a CALL:  call:NAME{args}
//     <|tool_response>50 <tool_response|>51   tool RESULT fed back
//
//  IMPORTANT: Apertura's OCTTokenizer does not match these markers as special tokens on ENCODE
//  (it BPE-splits "<|turn>" into 4 pieces), while HF maps the literal string to the single id 105.
//  So we build prompts at the TOKEN-ID level — splicing the control ids directly and tokenizing
//  only the surrounding text (role labels + content). This reproduces HF apply_chat_template
//  bit-for-bit (verified). DECODE is a direct id->string vocab lookup and is unaffected.
//
//  These ids are stable across the Gemma-4 family (one shared tokenizer); ESChatTokens lets a
//  caller override them if a future variant ever renumbers.
#include "ESTokenizer.h"
#include <string>
#include <vector>

namespace es {

struct ESChatTokens {
    int bos          = 2;
    int turnOpen     = 105, turnClose     = 106;  // turnClose is the generation STOP token
    int think        = 98;
    int channelOpen  = 100, channelClose  = 101;
    int toolOpen     = 46,  toolClose     = 47;
    int toolCallOpen = 48,  toolCallClose = 49;
    int toolRespOpen = 50,  toolRespClose = 51;
};

struct ESChatMessage {
    std::string role;      // "system" | "developer" | "user" | "assistant"
    std::string content;
};

struct ESToolCall {
    std::string name;      // function name from   call:NAME{...}
    std::string args;      // raw {key:val,...} body (Gemma's compact, non-JSON-quoted form)
};

struct ESParsedResponse {
    std::string             thought;     // text inside <|channel>thought ... <channel|>
    std::string             answer;      // visible answer (thinking + tool markup removed)
    std::vector<ESToolCall> toolCalls;
};

class ESChatTemplate {
public:
    explicit ESChatTemplate(const ESTokenizer & tok, ESChatTokens t = {}) : tok_(tok), t_(t) {}

    // Build prompt token ids, faithful to chat_template.jinja:
    //   - a leading system/developer message is hoisted into the system turn;
    //   - enableThinking injects <|think|> and leaves the model turn OPEN (model writes its own
    //     <|channel>thought ...); otherwise the model turn is pre-closed with an EMPTY thought
    //     channel, which suppresses reasoning;
    //   - addGenerationPrompt appends the trailing "<|turn>model\n..." so the model continues.
    std::vector<int> build(const std::vector<ESChatMessage> & messages,
                           bool enableThinking    = false,
                           bool addGenerationPrompt = true) const;

    // Split a generated id stream into thought / visible answer / tool calls.
    ESParsedResponse parse(const std::vector<int> & responseIds) const;

    int                  stopToken() const { return t_.turnClose; }
    const ESChatTokens & tokens()    const { return t_; }

private:
    std::vector<int> enc(const std::string & s) const { return tok_.encode(s, /*addSpecial=*/false); }
    const ESTokenizer & tok_;
    ESChatTokens        t_;
};

}  // namespace es
