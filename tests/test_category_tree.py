import pytest
from texttools.models import CategoryTree, CategoryNode


@pytest.fixture
def tree():
    tree = CategoryTree()
    tree.add_node("اخلاق", "root")
    tree.add_node("معرفت شناسی", "root")
    tree.add_node("متافیزیک", "root")
    tree.add_node("فلسفه ذهن", "root")
    tree.add_node("آگاهی", "فلسفه ذهن")
    tree.add_node("ذهن و بدن", "فلسفه ذهن")
    tree.add_node("امکان و ضرورت", "متافیزیک")
    tree.add_node("مغز و ترشحات", "ذهن و بدن")
    return tree


def test_level_count(tree):
    assert tree.get_max_depth() == 3


def test_none_node(tree):
    assert tree.get_node("سلامت") is None


def test_get_node(tree):
    assert isinstance(tree.get_node("آگاهی"), CategoryNode)


def test_add_duplicate_node(tree):
    with pytest.raises(ValueError, match="Cannot add آگاهی category twice"):
        tree.add_node("آگاهی", "root")


def test_wrong_parent(tree):
    with pytest.raises(ValueError, match="Parent category امکان not found"):
        tree.add_node("ضرورت", "امکان")


def test_remove_root(tree):
    with pytest.raises(ValueError, match="Cannot remove the root node"):
        tree.remove_node("root")


def test_remove_none(tree):
    with pytest.raises(ValueError, match="Category: ایجاب not found"):
        tree.remove_node("ایجاب")
