def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]