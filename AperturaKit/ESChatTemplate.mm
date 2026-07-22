#include "ESChatTemplate.h"

namespace es {

static std::string trimWS(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<int> ESChatTemplate::build(const std::vector<ESChatMessage> & messages,
                                       bool enableThinking, bool addGenerationPrompt,
                                       const std::vector<std::string> & toolDecls) const {
    std::vector<int> ids;
    auto put = [&](const std::vector<int> & v) { ids.insert(ids.end(), v.begin(), v.end()); };

    auto putQuoted = [&](const std::string & s) { put(encodeQuoted(s)); };

    ids.push_back(t_.bos);

    // System/Tool-definitions block: rendered when thinking is on OR tools are declared OR the
    // first message is a system/developer turn. (Gemma treats "developer" identically to "system".)
    bool hasSys = !messages.empty() &&
                  (messages[0].role == "system" || messages[0].role == "developer");
    size_t start = 0;
    if (enableThinking || hasSys || !toolDecls.empty()) {
        ids.push_back(t_.turnOpen); put(enc("system\n"));
        if (enableThinking) { ids.push_back(t_.think); put(enc("\n")); }   // <|think|> at the very top
        if (hasSys) { put(enc(trimWS(messages[0].content))); start = 1; }
        for (const auto & decl : toolDecls) {
            ids.push_back(t_.toolOpen); putQuoted(trimWS(decl)); ids.push_back(t_.toolClose);
        }
        ids.push_back(t_.turnClose); put(enc("\n"));
    }

    // Conversation turns. assistant -> "model". (Consecutive-assistant continuation merging is
    // not modelled here; normal alternating conversations are exact.)
    for (size_t i = start; i < messages.size(); ++i) {
        std::string role = messages[i].role == "assistant" ? "model" : messages[i].role;
        ids.push_back(t_.turnOpen); put(enc(role + "\n"));
        put(enc(trimWS(messages[i].content)));
        ids.push_back(t_.turnClose); put(enc("\n"));
    }

    // Generation prompt: open the model turn. Thinking-off pre-closes an empty thought channel
    // (reasoning suppressed); thinking-on leaves it open so the model writes its own.
    if (addGenerationPrompt) {
        ids.push_back(t_.turnOpen); put(enc("model\n"));
        if (!enableThinking) {
            ids.push_back(t_.channelOpen); put(enc("thought\n")); ids.push_back(t_.channelClose);
        }
    }
    return ids;
}

std::vector<int> ESChatTemplate::encodeQuoted(const std::string & s) const {
    static const std::string kQ = "<|\"|>";
    std::vector<int> ids;
    auto put = [&](const std::vector<int> & v) { ids.insert(ids.end(), v.begin(), v.end()); };
    size_t p = 0;
    for (size_t q; (q = s.find(kQ, p)) != std::string::npos; p = q + kQ.size()) {
        if (q > p) put(enc(s.substr(p, q - p)));
        ids.push_back(t_.quote);
    }
    if (p < s.size()) put(enc(s.substr(p)));
    return ids;
}

ESParsedResponse ESChatTemplate::parse(const std::vector<int> & ids) const {
    ESParsedResponse r;
    std::vector<int> answerIds;
    const size_t n = ids.size();

    for (size_t i = 0; i < n; ) {
        int id = ids[i];

        if (id == t_.channelOpen) {
            // <|channel>{label}\n {body} <channel|>  -- e.g. label "thought".
            std::vector<int> seg; ++i;
            while (i < n && ids[i] != t_.channelClose) seg.push_back(ids[i++]);
            if (i < n) ++i;  // consume <channel|>
            std::string s = tok_.decode(seg, /*skipSpecial=*/true);
            size_t nl = s.find('\n');
            std::string body = (nl == std::string::npos) ? std::string() : s.substr(nl + 1);
            std::string piece = trimWS(body);
            if (!piece.empty()) r.thought += (r.thought.empty() ? "" : "\n") + piece;

        } else if (id == t_.toolCallOpen) {
            // <|tool_call>call:NAME{args}<tool_call|>
            std::vector<int> seg; ++i;
            while (i < n && ids[i] != t_.toolCallClose) seg.push_back(ids[i++]);
            if (i < n) ++i;  // consume <tool_call|>
            std::string s = trimWS(tok_.decode(seg, /*skipSpecial=*/true));
            size_t cpos = s.find("call:");
            if (cpos != std::string::npos) s = s.substr(cpos + 5);
            ESToolCall tc;
            size_t brace = s.find('{');
            if (brace != std::string::npos) {
                tc.name = trimWS(s.substr(0, brace));
                size_t close = s.rfind('}');
                tc.args = (close != std::string::npos && close > brace)
                              ? s.substr(brace + 1, close - brace - 1)
                              : s.substr(brace + 1);
            } else {
                tc.name = trimWS(s);
            }
            r.toolCalls.push_back(tc);

        } else if (id == t_.turnOpen || id == t_.turnClose || id == t_.think ||
                   id == t_.channelClose || id == t_.toolCallClose ||
                   id == t_.toolOpen || id == t_.toolClose ||
                   id == t_.toolRespOpen || id == t_.toolRespClose || id == t_.quote) {
            ++i;  // stray control token — skip

        } else {
            answerIds.push_back(id);
            ++i;
        }
    }

    r.answer = trimWS(tok_.decode(answerIds, /*skipSpecial=*/true));
    return r;
}

}  // namespace es
