"""Read-only geometry view with the complete actual depth-observation history."""


class AxisHistoryViewV16:
    def __init__(self, mapper, frames):
        self._mapper = mapper
        self.keyframes = tuple(frames)
        if not self.keyframes or any(a.timestamp_s >= b.timestamp_s
                                     for a, b in zip(self.keyframes, self.keyframes[1:])):
            raise ValueError('complete chronological actual observations required')

    def __getattr__(self, name):
        return getattr(self._mapper, name)
