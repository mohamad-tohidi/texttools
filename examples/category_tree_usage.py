import json
from texttools import CategoryTree

# Create a CategoryTree from scratch
tree = CategoryTree()
tree.add_node("اخلاق", "root")
tree.add_node("معرفت شناسی", "root")
tree.add_node("متافیزیک", "root", description="اراده قدرت در حیطه متافیزیک است")
tree.add_node("فلسفه ذهن", "root")
tree.add_node("آگاهی", "فلسفه ذهن")
tree.add_node("ذهن و بدن", "فلسفه ذهن")
tree.add_node("امکان و ضرورت", "متافیزیک")

# Print your tree
print(tree.dump_tree())

# Create a CategoryTree from a json file
file_path = "your_json_file_path"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

tree = CategoryTree.from_dict(root=data)

# Print your tree
print(tree.dump_tree())
