def find_project_root(current_path):
    for parent in current_path.parents:
        if (parent / '.git').exists() or (parent / 'src').exists():
            return parent
    return None 