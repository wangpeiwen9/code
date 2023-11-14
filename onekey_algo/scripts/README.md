# ShellScripts
<div  align="center">    
<img src="./assets/scripts.png" width = "40%"/>
</div>

All useful scripts is merge into Scripts. The module is built for new employee 
or anyone in need. 

**Python3.6+ is required**

We recommend you load or store data in CSV format.

## MailStone
>2020-05-16 Scripts for learning to rank, reuse of ShellScripts.

## Install

1. Install requirements

	```bash
	$ pip install -r requirements.txt --upgrade
	```

	>If some software slow down your installation

	```bash
	$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --upgrade
	```

2. Modify [config](config) for your own configuration.

    ```python
    import os
   
   
    class Config:
        # Where to save scripts.
        SCRIPTS_ROOT = 'bin'
        # Scripts' template
        SCRIPTS_TEMP = './config/scripts_with_argparse.onekey_algo'
        # Manual template
        MAN_TEMP = './config/man.onekey_algo'
        # Which module to generate as scripts.
        MODULE_DIRs = [os.path.join(os.getcwd(), 'core')]
        # PY_INTERPRETER
        PY_INTERPRETER = None
        # Learning to rank HOME
        ONEKEY_HOME = os.path.dirname(os.path.dirname(os.getcwd()))


    ```
	* `SCRIPTS_ROOT`: Where to save scripts. _Recommend don't edit_. 
	* `SCRIPTS_TEMP`: Template for scripts. _Recommend don't edit_. 
	* `MAN_TEMP`: Manuel template for scripts. _Recommended don't edit_. 
	* `MODULE_DIRs`: Which module to be converted to scripts. _Recommend keep `os.path.join(os.getcwd(), 'core')`_. 
	* `PY_INTERPRETER`: Python interpreter which you can specify, default is current `python` interpreter. 
    * `ONEKEY_HOME`: Learning to rank home, _Recommended don't edit_.
    
2. Install ShellScripts

	```bash
	$ cd ShellScripts
	$ python setup.py -i path_to_your_interpreter
	```
	
	`path_to_your_interpreter` is optical, default is your activated python interpreter.

## Usage

ShellScripts is easy to use. Just `man_lrt` to get all [AVAILABLE](./doc/manuel.md) scripts and its usage. 



Use `some_scripts -h` to get usage of specific scripts.

### Using man_ltr
Approximate string matching of supported scripts.

```bash
$ man_ltr get
```

This will show all scripts that `get` in its name.

### Using specific scripts

eg. `get_tensor_in_ckpt` 

```bash
$ dataset_sample_distribution -h
usage: dataset_sample_distribution [-h] --py_config PY_CONFIG
                                   [--subset SUBSET]
                                   [--feature_id [FEATURE_ID [FEATURE_ID ...]]]
                                   [--resolution [RESOLUTION [RESOLUTION ...]]]
                                   [--save_dir SAVE_DIR] [--ignore_unused]
                                   [--random_sample_ratio RANDOM_SAMPLE_RATIO]

Analysis dataset positive and negative distribution.

optional arguments:
  -h, --help            show this help message and exit
  --py_config PY_CONFIG
                        Python config file. REQUIRED!
  --subset SUBSET       Subset of data to be used. Default None for config
                        doesn\'t has subset.
  --feature_id [FEATURE_ID [FEATURE_ID ...]]
                        Feature id used to calculate reverse order.
  --resolution [RESOLUTION [RESOLUTION ...]]
                        Resolution used to count values. negative value means
                        // 10^x.
  --save_dir SAVE_DIR   Where to save pictures of distribution.
  --ignore_unused       Ignore `index2use`, `force2value` settings in data
                        config. Default False.
  --random_sample_ratio RANDOM_SAMPLE_RATIO
                        Random sample ratio.
```

### Use it!
```bash
$ feature_distribution  --input_file path_to_your_file
```

#### REQUIRED!

Helper information red blink means this parameter is required.

#### PIPE INPUT SUPPORTED!

Helper information green blink means this parameter is can use stdin as input.

Like the following shows.
```bash
cat path_to_your_file |feature_distribution
```

## Contribute

Just create function in `core` module as you want. But we also have suggest template!

```python
__all__ = ["some_func"]

def some_func(**kwargs):
    """
    Some comments...
    
    :param some_parameters: it's definition.
    :param pipe_supported_parameters: Add {PIPE_HINT} indicates this parameter support pipe as input. 
    """
    # Do what you want!
    pass
```

### `__all__`

This variable stores which function should be exported as scripts. `setup.py` will ignore module which is not in `__all__`.

This is also suit for `core.__init__.__all__`. When you add new python file, file name should all be add.

```python
__all__ = ['about_validing_data', 'about_tensorflow', 'your_new_py_file']
```