from htmltools import tags, div, css
from itertools import combinations


def input_return_text_area(id, label, value="", rows=1, placeholder=""):
    classes = ["form-control", "returnTextArea"]
    area = tags.textarea(
        value,
        id=id,
        class_=" ".join(classes),
        style=css(width="100%", height=None, resize="vertical"),
        placeholder=placeholder,
        rows=rows,
        cols=None,
        autocomplete=False,
        spellcheck=False,
    )

    def shiny_input_label(id, label=None):
        cls = "control-label" + ("" if label else " shiny-label-null")
        return tags.label(label, class_=cls, id=id + "-label", for_=id)

    return div(
        shiny_input_label(id, label),
        area,
        None,
        class_="form-group shiny-input-container",
        style=css(width="100%"),
    )


def is_empty(x):
    return x is None or all(c.isspace() for c in x)


def qterms(vars, nway=2):
    return [f"I({v} ** {p})" for p in range(2, nway + 1) for v in vars]


def iterms(vars, nway=2, sep=":"):
    cvars = list(combinations(vars, 2))
    if nway > 2:
        cvars += list(combinations(vars, nway))
    return [f"{sep}".join(c) for c in cvars]
