
# Build Instructions

- For python, either pip install pyyaml, or activate the virtual environment first. I like using uv
`uv venv`, then `source .venv/bin/activate`
- `mkdir build && cd build`
- `cmake -G "Ninja Multi-Config" ../`
- `ninja`

