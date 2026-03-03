from core.registry import get_by_category
for cat, gens in get_by_category().items():
    for g in gens:
        has_color = any(p.type == chr(99)+chr(111)+chr(108)+chr(111)+chr(114) for p in g.params)
        print(has_color, cat, g.label)
