class WelchWindowError(Exception):
    def __init__(self, winsize):
        self.winsize = winsize

    def __str__(self):
        return f'The welch window size ({self.winsize}) is too small for current bands, consider increasing the window size.'


class SignalTooShortError(Exception):
    def __init__(self, siglen):
        self.siglen = siglen

    def __str__(self):
        return f'Current signal length ({self.siglen}) is too short for the welch size and this method will not work.'
