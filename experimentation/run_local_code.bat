:: Need to run this in YOUR_PROJECT_DIR
:: Set the values in .env.examples and change it to .env
:: Refer to https://github.com/microsoft/MLOpsPython/blob/master/docs/development_setup.md
set PYTHONPATH=%PYTHONPATH%;YOUR_PROJECT_DIR
echo %PYTHONPATH%
python ml_service/pipelines/img_class_build_train_pipeline.py && python ml_service/pipelines/run_train_pipeline.py