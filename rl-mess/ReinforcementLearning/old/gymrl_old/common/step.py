class Step:
    def __init__(self, **kwargs):
        self.reward = kwargs.get('reward')
        self.state = kwargs.get('state')
        self.action = kwargs.get('action')

    def __repr__(self):
        return f'<Step(reward={self.reward} state={self.state} action={self.action})>'
