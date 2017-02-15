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

if [[ ! -d venv ]]; then
	virtualenv venv
else
	echo Virtualenv existiert bereits.
fi

if [[ -r venv/bin/activate ]]; then
	source venv/bin/activate

	if [[ "$(pip freeze | grep gensim)" == "" ]]; then
		pip install gensim
	else
		if confirm "Gensim is already installed. Upgrade? (yN)" N; then
			pip install --upgrade gensim
		fi
	fi

else
	echo Cannot source virtualenv activation file!
	exit 1
fi
