"""Resolution-aware touch controls and stable detection colors."""

import colorsys


def layout_metrics(width, height):
    font = max(18, min(32, round(min(width, height) / 44)))
    return dict(
        font=font, touch=max(48, font * 2), side=font * 15, portrait=width < height or width < 1000
    )


def category_color(label):
    # Stable across frames/processes; never use Python's randomized hash().
    from scenes.yolox import LABELS

    index = LABELS.index(label) if label in LABELS else 0
    return tuple(
        round(c * 255) for c in colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.65, 1.0)
    )


def stylesheet(font, touch):
    return f"""
        window {{ background: #101820; color: #ecf3f8; }}
        label, entry, treeview, menuitem {{ font-size: {font}px; }}
        button, combobox {{ font-size: {font}px; }}
        button, entry {{ min-height: {touch - 16}px; padding: 8px 12px; }}
        button {{ background-image: none; background-color: #e8eef4; color: #14202b;
                  border-radius: 8px; }}
        button label, combobox cellview {{ color: #14202b; }}
        button:disabled {{ background-color: #8795a3; }}
        menu, menuitem, menuitem label, menuitem cellview,
        popover, list, row {{ background-color: #e8eef4; color: #14202b; }}
        menuitem:hover, menuitem:hover label, menuitem:hover cellview,
        row:selected {{ background-color: #c0d9ee; color: #14202b; }}
        .title {{ font-size: {round(font * 1.25)}px; font-weight: bold; }}
        .input-mode:checked, .primary {{ background-color: #b9dce9; color: #10212e; }}
        .muted {{ color: #afc0cc; font-size: {max(12, round(font * 0.65))}px; }}
    """.encode()
