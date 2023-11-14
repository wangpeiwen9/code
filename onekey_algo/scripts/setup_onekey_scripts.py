# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/8/16
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import argparse
import importlib
import os
import platform
from inspect import signature

from scripts.config.config import Config
from utils.about_log import ColorPrinter, PIPE_HINT, REQUIRED
from utils.common import create_dir_if_not_exists
import platform

parser = argparse.ArgumentParser(description='Setup for Shell Scripts.')

parser.add_argument('-i', '--interpreter', dest='interpreter', default=None,
                    help='Specific python interpreter to be used.')
args = parser.parse_args()
# Where to save scripts.
SCRIPTS_ROOT = Config.SCRIPTS_ROOT
# Which module to generate as scripts.
MODULE_DIRs = Config.MODULE_DIRs
# LTR HOME.
ONEKEY_HOME = Config.ONEKEY_HOME
# Python interpreter.
PY_INTERPRETER = args.interpreter or Config.PY_INTERPRETER or os.popen('which python').read().strip()
print('Use PY_INTERPRETER: %s' % PY_INTERPRETER)
# Add this software to PYTHONPATH and SCRIPTS_ROOT to PATH

if platform.system() == 'Windows':
    pass
else:
    bashrc = [l.strip() for l in os.popen('cat ~/.bashrc').readlines()]
    # If PATH already exists, continue.
    if not bashrc or "export PYTHONPATH=%s:$PYTHONPATH" % ONEKEY_HOME not in bashrc:
        os.system("echo 'export PYTHONPATH=%s:$PYTHONPATH' >> ~/.bashrc" % ONEKEY_HOME)
    if not bashrc or "export PATH=%s:$PATH" % os.path.join(os.getcwd(), SCRIPTS_ROOT) not in bashrc:
        os.system("echo 'export PATH=%s:$PATH' >> ~/.bashrc" % os.path.join(os.getcwd(), SCRIPTS_ROOT))

create_dir_if_not_exists(SCRIPTS_ROOT)


def _find_module(module_dir_):
    # Find module which can be exported as scripts.
    if module_dir_ and os.path.exists(module_dir_) and os.path.isdir(module_dir_):
        return [".%s" % os.path.splitext(os.path.basename(f_))[0] for f_ in os.listdir(module_dir_)
                if os.path.isfile(os.path.join(module_dir_, f_)) and f_.endswith('.py') and not f_.startswith('_')]


def _convert_params_to_argparse(func):
    doc_chunk = func.__doc__ or ''
    # Replace some special char.
    doc_chunk = doc_chunk.replace("'", "\\'")
    doc_chunk = doc_chunk.replace('%', '%%')

    doc_lines = doc_chunk.split('\n')
    func_description = []
    # Get function description.
    for l in doc_lines:
        if 'Args:' in l or 'Returns:' in l:
            break
        func_description.append(l.rstrip())
    func_description = '\\n'.join(func_description)
    args_list = ["parser = argparse.ArgumentParser(description='{}')".format(func_description)]
    argparse_tmp = """parser.add_argument('--{param_name}', {others}, help=r"{param_help}")"""

    def __judge_type(anno: str):
        if 'int' in anno:
            return 'type=int'
        elif 'float' in anno:
            return 'type=float'
        elif 'bool' in anno:
            return 'type=bool'
        else:
            return 'type=str'

    sig = signature(func)
    for k in sig.parameters:
        annotation = str(sig.parameters[k].annotation)
        is_list = "nargs='*'" if 'List' in annotation else ''
        param_type = __judge_type(annotation)

        # If str and not list, change default to ''.
        param_default = "default=None, required=True"
        if 'empty' not in str(sig.parameters[k].default):
            param_default = "default=%s" % str(sig.parameters[k].default)
            if 'str' in param_type and not is_list and sig.parameters[k].default is not None:
                param_default = "default='%s'" % str(sig.parameters[k].default)

        # If bool in parameter.
        param_action = ''
        if 'bool' in param_type:
            param_action = "action='store_true'"
            if 'True' == str(sig.parameters[k].default):
                param_action = "action='store_false'"
            param_type = ''

        # Get help for each params.
        param_help = 'No found!'
        param_choice = ''
        for line in doc_lines:
            if "        %s:" % k in line:
                param_help = line.replace("        %s:" % k, '').strip()
                # Replace PIPE_HINT and REQUIRED to colored message.
                param_help = param_help.replace('{PIPE_HINT}', PIPE_HINT)
                param_help = param_help.replace('{REQUIRED}', REQUIRED)
                try:
                    choice_start_idx = param_help.index('{CHOICE}') + len('{CHOICE}')
                    param_choice = [c_.strip() for c_ in param_help[choice_start_idx:].strip(': .').split(',')]
                    param_choice = f"choices={param_choice}"
                except ValueError:
                    pass
                break
        args_list.append(argparse_tmp.format(param_name=k,
                                             param_help=param_help,
                                             others=', '.join(a for a in [is_list,
                                                                          param_default,
                                                                          param_type,
                                                                          param_action,
                                                                          param_choice] if a)))
    return '\n'.join(args_list)


# Firstly get scripts template.
scripts_tmp = open(Config.SCRIPTS_TEMP, encoding='utf-8').read()
already_has_name = {}
manss = {}
cp = ColorPrinter()
# Save scripts into directory.
error_modules = []
for MODULE_DIR in MODULE_DIRs:
    modules = _find_module(MODULE_DIR)
    if not modules:
        cp.cprint("In %s doesn't find any module!" % MODULE_DIR, 'yellow')
        continue
    module_dir = MODULE_DIR.replace(ONEKEY_HOME, '').replace('/', '.').replace('\\', '.').lstrip('.')
    importlib.import_module(module_dir)
    for m in modules:
        # If can't import module continue.
        try:
            module = importlib.import_module(m, module_dir)
            if hasattr(module, '__all__'):
                for attr_name in module.__all__:
                    # Generate scripts content first.
                    attr = getattr(module, attr_name)
                    attr_desc = "{}\n{}".format('|{c} Function {func} Definition {c}|'
                                                .format(c="-" * 30, func=cp.color_text(attr_name.center(25), 'cyan')),
                                                attr.__doc__)
                    attr_source = "%s%s.%s" % (module_dir, m, attr_name)
                    attr_import = "from %s%s import %s" % (module_dir, m, attr_name)
                    script = scripts_tmp.format(source_module=attr_source,
                                                import_func=attr_import,
                                                argparse_description=_convert_params_to_argparse(attr))
                    # Get saving file name.
                    if attr_name in already_has_name:
                        already_has_name[attr_name] += 1
                    else:
                        already_has_name[attr_name] = 0
                    save_filename = attr_name + (
                        '%d' % already_has_name[attr_name] if already_has_name[attr_name] else '')

                    color_attr_source = cp.color_text('%s' % attr_source.center(96), 'cyan')
                    color_save_filename = cp.color_text('%s' % save_filename.center(48), 'magenta')
                    if already_has_name[attr_name]:
                        warning = cp.color_text('WARNING: Scripts %s is already exists! ', 'yellow')
                        print('%s\nSave %s to %s' % (warning, color_attr_source, color_save_filename))
                    else:
                        print('Creating script %-64s to %s' % (color_attr_source, color_save_filename))
                    with open(os.path.join(SCRIPTS_ROOT, save_filename), 'w', encoding='utf-8') as f:
                        f.write("#! %s\n%s" % (PY_INTERPRETER, script))

                    # Add execute to scripts.
                    if platform.system() == 'Linux':
                        os.system('chmod u+x %s' % os.path.join(SCRIPTS_ROOT, save_filename))
                    manss[save_filename] = attr_desc
        except Exception as e:
            error_modules.append((module_dir, m, e))

if error_modules:
    cp.cprint('Create scripts failed modules:\n%s' %
              '\n'.join(['\t%s%s because of %s' % (d, m, e) for d, m, e in error_modules]), 'yellow')

# Saving manuals of Shell Scripts
man_tmp = open(Config.MAN_TEMP, encoding='utf-8').read()
with open(os.path.join(SCRIPTS_ROOT, 'man_onekey'), 'w', encoding='utf-8') as f:
    f.write("#! %s\n%s" % (PY_INTERPRETER, man_tmp.format(func_description=manss)))

if platform.system() == 'Linux':
    os.system('chmod u+x %s' % (os.path.join(SCRIPTS_ROOT, 'man_onekey')))
    print(cp.color_text('Enjoy onekey_algo scripts using: source ~/.bashrc', 'green', 'blink'))
