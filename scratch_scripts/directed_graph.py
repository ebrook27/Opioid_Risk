### 3/17/26, EB: drawing causal graph

import graphviz

known_features = [
    'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding',
    'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes',
    'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle',
    'Single-Parent Household', 'Unemployment', 'Uninsured', 'RX Rate',  
]

hidden_features = [
    'Illegal Supply',   
]


known_edges = [
    ("Below Poverty", "Unemployment"),
    ("No High School Diploma", "Unemployment"),
    ("No Vehicle", "Unemployment"),
    
    ("Below Poverty", "Uninsured"),
    ("No High School Diploma", "Uninsured"),
    ("17 or Younger", "Uninsured"),
    ("65 or Older", "Uninsured"),
    ("Single-Parent Household", "Uninsured"),
    
    ("17 or Younger", "RX Rate"),
    ("65 or Older", "RX Rate"),

    
    
    # ("Unemployment", "RX Rate"),
    # ("Uninsured", "RX Rate"),
    # ("No High School Diploma", "Unemployment"),
    ("RX Rate", "Mortality"),
    ("Unemployment", "Mortality"),
    ("Uninsured", "Mortality"),
    ("Unemployment", "Uninsured"),
]

hidden_edges = [
    ("Illegal Supply", "RX Rate"),
    ("RX Rate", "Illegal Supply"),
    ("Illegal Supply", "Mortality"),
]

g = graphviz.Digraph()
for name in known_features:
    g.node(name, fontsize="10")

for name in hidden_features:
    g.node(name, style="dashed", fontsize="10")


g.node("Mortality", style="filled", fontsize="10")


# for edge in known_edges:
#     g.edge(edge[0], edge[1])

for src, dest in known_edges:
    g.edge(src, dest)

for src, dest in hidden_edges:
    g.edge(src, dest, style="dashed")

g.view()