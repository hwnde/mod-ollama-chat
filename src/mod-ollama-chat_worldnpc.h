#ifndef MOD_OLLAMA_CHAT_WORLDNPC_H
#define MOD_OLLAMA_CHAT_WORLDNPC_H

#include "ScriptMgr.h"

class OllamaWorldNpcChatter : public WorldScript
{
public:
    OllamaWorldNpcChatter();
    void OnUpdate(uint32 diff) override;

private:
    void HandleNpcProximityChatter();
};

#endif // MOD_OLLAMA_CHAT_WORLDNPC_H
