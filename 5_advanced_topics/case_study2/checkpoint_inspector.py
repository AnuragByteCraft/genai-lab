def print_checkpoint_info(history: list, config: dict):
    print(f"🧠 {len(history)} checkpoints - thread: {config['configurable']['thread_id']}")

    for i, snapshot in enumerate(reversed(history)):
        print(f"\n📍 #{i + 1} → {snapshot.next or 'END'}")
        messages = snapshot.values.get('messages', [])

        for msg in messages[-2:]:
            role = getattr(msg, 'role', '?')
            content = str(getattr(msg, 'content', ''))[:50]
            print(f"  [{role}] {content}...")

            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get('name', 'unknown') if isinstance(tc, dict) else str(tc)
                    args = tc.get('args', {}) if isinstance(tc, dict) else {}
                    print(f"    🛠️  {name}({args})")