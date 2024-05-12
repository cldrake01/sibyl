class SignatureError(Exception):
    def __init__(self):
        self.message = "The provided callable must have a Config object as an argument."
        super().__init__(self.message)
