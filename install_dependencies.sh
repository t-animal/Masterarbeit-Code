#!/bin/bash

set -e

confirm() {
    while true; do
        read -p "$1 " yn
        case ${yn:-$2} in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}


install_package() {
	version=$(pip freeze | grep "$1" | cut -d"=" -f 3)
	upstream=$(wget -qO- https://pypi.python.org/pypi/$1/json | grep -E ' {8}"[0-9."]*": \[' | sort -V | tail -n 1 | tr -d ' ":[')
	if [[ "$version" == "" ]]; then
		pip install "$1"
	else
		if confirm "$1 (v$version) is already installed. Upgrade (upstream is v$upstream)? (yN)" N; then
			pip install --upgrade "$1"
		fi
	fi
}

if [[ ! -d venv ]]; then
	virtualenv -p python3 --system-site-packages venv
else
	echo Virtualenv existiert bereits.
fi

if [[ -r venv/bin/activate ]]; then
	source venv/bin/activate

	#can't hurt to upgrade pip first
	pip install --upgrade pip

	install_package gensim
	install_package scikit-learn
	install_package matplotlib
	install_package argcomplete
	install_package pylatex

	echo "=============================================================================="
	echo "To use completion for tester.py, execute this after activating the virtualenv:"
	echo 'eval "$(register-python-argcomplete ./tester.py)"'
	echo "=============================================================================="

else
	echo Cannot source virtualenv activation file!
	exit 1
fi
