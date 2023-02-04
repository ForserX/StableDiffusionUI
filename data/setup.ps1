Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
Write-Host "Installing Build Dependencies"
python -m venv .\repo\shark.venv\
.\repo\shark.venv\Scripts\activate
pip install -r requirements_shark.txt
pip install --pre torch-mlir torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://llvm.github.io/torch-mlir/package-index/
pip install --upgrade -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html iree-compiler iree-runtime
Write-Host "Building SHARK..."
pip install -e . -f https://llvm.github.io/torch-mlir/package-index/ -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html
deactivate

Write-Host "Building ONNX..."
python -m venv .\repo\onnx.venv\
.\repo\onnx.venv\Scripts\activate
pip install -r requirements_onnx.txt