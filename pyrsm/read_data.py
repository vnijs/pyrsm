import importlib.resources

def greet(recipient):
    template = importlib.resources.read_text("data.data", "testing.txt")
    return template.format(recipient=recipient)