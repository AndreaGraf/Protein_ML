"""Automates Python scripts formatting, linting and Mkdocs documentation."""

import ast
import importlib
import re
import json, yaml
from collections import defaultdict
from pathlib import Path
from typing import Union, get_type_hints
import os
import shutil

from mkgendocs.gendocs import generate
from mkgendocs.parse import GoogleDocString

import re

def format_md_text(text):
    """Formats the text in the docstring to markdown"""
    lines = text.split('\n')
    python_lines = []

    for i,line in enumerate(lines):
        #cleanup lines
        line = line.strip()
        line = re.sub(r'\s+', ' ', line)
        
        if re.match(r'^python(3)? ', line):
            line = f'```python\n{line}\n```'
            print(line)
        lines[i] = line
    
    return '\n'.join(lines)
        

def data_to_markdown(data):
    """Converts the data from the docstring parser to markdown"""
    markdown = []
    for d in data:
        if 'header' in d and len(d['header']) > 0:
            markdown.append(f'### {d["header"]}')
        if 'signature' in d:
            markdown.append(f'```python\n {d["signature"]}\n```')
        if 'description' in d:
            markdown.append(d['description'])
        if 'args' in d and len(d['args']) > 0:
            markdown.append('### Arguments')
            for arg in d['args']:
                markdown.append(f'* **{arg["name"]}** ({arg["type"]}): {arg["description"]}')
        if 'returns' in d:
            markdown.append('### Returns')
            markdown.append(f'* **{d["returns"]["type"]}**: {d["returns"]["description"]}')
        if 'raises' in d:
            markdown.append('### Raises')
            for r in d['raises']:
                markdown.append(f'* **{r["type"]}**: {r["description"]}')
        if 'attributes' in d:
            markdown.append('### Attributes')
            for attr in d['attributes']:
                markdown.append(f'* **{attr["name"]}** ({attr["type"]}): {attr["description"]}')
        if 'text' in d:
            markdown.append(format_md_text(d['text']))

    return '\n'.join(markdown)



def add_val(indices, value, data):
    if not len(indices):
        return
    element = data
    for index in indices[:-1]:
        element = element[index]
    element[indices[-1]] = value



def add_mkgen_pages( mkgendocs_f: str, repo_dir: Path, match_string: str, insert_string:str) -> str:
    """function to add pages structure to mkgendocs.yml
    Args:   
        mkgendocs_f (str): mkgendocs.yml configuration file 
        repo_dir (pathlib.Path): path to yaml file        
    """
    with open(f'{repo_dir}/{mkgendocs_f}', 'r+') as mkgen_config:
        contents = mkgen_config.readlines()
        if match_string in contents[-1]:
            contents.append(insert_string)
        else:

            for index, line in enumerate(contents):
                if match_string in line and insert_string not in contents[index + 1]:

                    contents = contents[: index + 1]
                    contents.append(insert_string)
                    break

    return contents


def automate_mkdocs_from_docstring(
    mkdocs_dir: Union[str, Path], mkgendocs_f: str, repo_dir: Path, match_string: str, templ_dir: str = 'templates'
) -> dict:
    """Automates the -pages for mkgendocs package by adding all Python functions in a directory to the mkgendocs config.
    Args:
        mkdocs_dir (typing.Union[str, pathlib.Path]): directory for navigation in Mkdocs
        mkgendocs_f (str): The configurations file for the mkgendocs package
        repo_dir (pathlib.Path): directory to search in for Python functions in
        match_string (str): the text to be matches, after which the functions will be added in mkgendocs format
    Example:
        >>>
        >>> automate_mkdocs_from_docstring('scripts', repo_dir=Path.cwd(), match_string='pages:')
    Returns:
        list: list of created markdown files and their relative paths

    """
    p = repo_dir.glob('**/*.py')
    scripts = [x for x in p if x.is_file()]

    if Path.cwd() != repo_dir:  # look for mkgendocs.yml in the parent file if a subdirectory is used
        repo_dir = repo_dir.parent

    functions = defaultdict(dict)
    intro_contents = defaultdict(dict)
    structure = fix(defaultdict)()
    full_repo_dir = f"{repo_dir}/"
    for script in scripts:

        with open(script, 'r') as source:
            tree = ast.parse(source.read())
        funcs = {
        "classes":[],
        "functions":[]
        }
        intros = []
        for child in ast.iter_child_nodes(tree):
            try:
                if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    if child.name not in ['main']:

                        relative_path = str(script).replace(full_repo_dir, "").replace("/", ".").replace(".py", "")
                        module = importlib.import_module(relative_path)
                        f_ = getattr(module, child.name)
                        function = f_.__name__
                        if isinstance(child, (ast.ClassDef)):
                            funcs["classes"].append(function)
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            funcs["functions"].append(function)
                elif isinstance(child, (ast.Expr)):
                    child.value.s
                    intros.append(child.value.s)
    

            except Exception as e:
                print(f"trouble on importing {script.stem}")
                print(f"did not document {child.name}")
                print(e)
        if not funcs["classes"]:
            funcs.pop("classes")
        if not funcs["functions"]:
            funcs.pop("functions")
        if funcs and script.parts[-1] not in ['automate_mkdocs.py', '__init__.py']:
            functions[script] = funcs
            if intros:
                intro_contents[script] = '\n'.join(intros) 
        elif  len(intros)>0 and script.parts[-1] == '__init__.py':
            script_new = Path(*list(script.parts[:-1]) + ['index.py'])
            intro_contents[script_new] = '\n'.join(intros)
            functions[script_new] = {}

    # create template files with intros
    for path, intro in intro_contents.items():

        relative_path = str(path).replace(full_repo_dir, "").replace(".py", "")
        templ_path = f'{repo_dir}/{templ_dir}/modules'
        for p in relative_path.split("/")[:-1]:
            if not os.path.exists(f'{templ_path}/{p}'):
                os.makedirs(f'{templ_path}/{p}')
            templ_path = f'{templ_path}/{p}'

        #format the module intro dioctsrings
        docstring_parser = GoogleDocString(intro)
        data = docstring_parser.parse()
        markdown_str = data_to_markdown(data)
            
        with open(f'{repo_dir}/{templ_dir}/modules/{relative_path}.md', 'w') as f:
            f.write(f'{markdown_str}\n\n')
            f.write('\n{{autogenerated}}')
        f.close()

    #create insert_string to add the pages
    insert_string = ''
    for path, function_names in functions.items():
        relative_path = str(path).replace(full_repo_dir, "").replace(".py", "")
        insert_string += (
            f'  - page: "{mkdocs_dir}/{relative_path}.md"\n    '
        )
        if 'index' not in relative_path:
            insert_string += (
                f'source: "{relative_path}.py"\n' 
            )
        else:
            insert_string += (
                f'source: \"{relative_path.replace("index","__init__")}.py\"\n' 
            )
        page = f"{mkdocs_dir}/{relative_path}"
        split_page = page.split("/")
        split_page = [f"  - {s}" for s in split_page]
        page = f"{page}.md"

        add_val(split_page, page, structure)
        for class_name, class_list in function_names.items():
            insert_string += f'    {class_name}:\n'
            f_string = ''
            for f in class_list:
                insert_f_string = f'      - {f}\n'
                f_string += insert_f_string

            insert_string += f_string
        insert_string += "\n"

    contents = add_mkgen_pages( mkgendocs_f, repo_dir, match_string, insert_string)


    with open(f'{repo_dir}/{mkgendocs_f}', 'w') as mkgen_config:
        mkgen_config.writelines(contents)

    return structure


def automate_nav_structure(
    mkdocs_dir: Union[str, Path], mkdocs_f: str, repo_dir: Path, match_string: str, structure: dict
) -> str:
    """Automates the -pages for mkgendocs package by adding all Python functions in a directory to the mkgendocs config.
    Args:
        mkdocs_dir (typing.Union[str, pathlib.Path]): textual directory for the hierarchical directory & navigation in Mkdocs
        mkgendocs_f (str): The configurations file for the mkgendocs package
        repo_dir (pathlib.Path): textual directory to search for Python functions in
        match_string (str): the text to be matches, after which the functions will be added in mkgendocs format
    Example:
        >>>
        >>> automate_mkdocs_from_docstring('scripts', repo_dir=Path.cwd(), match_string='pages:')
    Returns:
        str: feedback message

    """
    insert_string = yaml.safe_dump(json.loads(json.dumps(structure, indent=4))).replace("'", "")
    # print(structure)
    with open(f'{repo_dir}/{mkdocs_f}', 'r+') as mkgen_config:
        contents = mkgen_config.readlines()
        if match_string in contents[-1]:
            contents.append(insert_string)
        else:

            for index, line in enumerate(contents):
                if match_string in line and insert_string not in contents[index + 1]:

                    contents = contents[: index + 1]
                    contents.append(insert_string)
                    break

    with open(f'{repo_dir}/{mkdocs_f}', 'w') as mkgen_config:
        mkgen_config.writelines(contents)


def fix(f):
    """Allows creation of arbitrary length dict item

    Args:
        f (type): Description of parameter `f`.

    Returns:
        type: Description of returned object.

    """
    return lambda *args, **kwargs: f(fix(f), *args, **kwargs)



def indent(string: str) -> int:
    """Count the indentation in whitespace characters.
    Args:
        string (str): text with indents
    Returns:
        int: Number of whitespace indentations

    """
    return sum(4 if char == '\t' else 1 for char in string[: -len(string.lstrip())])




def main():
    """Execute when running this script."""
    python_tips_dir = Path.cwd().joinpath('')

    structure = automate_mkdocs_from_docstring(
       mkdocs_dir='modules',
       mkgendocs_f='mkgendocs.yml',
       repo_dir=python_tips_dir,
       match_string='pages:\n',
    )

    automate_nav_structure(
       mkdocs_dir='modules',
       mkdocs_f='mkdocs.yml',
       repo_dir=python_tips_dir,
       match_string='- Home: index.md\n',
       structure=structure
    )

    # copy  indexfile to temporary location as gendocs will remove it
    if not os.path.exists('temp'):
       os.mkdir('temp')
    shutil.copyfile('docs/index.md', 'temp/index.md.tmp')
    generate('mkgendocs.yml')
    
    # redo the index page to not eqal the README
    os.remove('docs/index.md')
    shutil.move('temp/index.md.tmp', 'docs/index.md')
    os.rmdir('temp')



if __name__ == '__main__':
    main()
