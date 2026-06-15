POWERSHELL SCRIPT TO RENDER lambda_package.zip


& {
$PackageDir = "lambda_package"
$ZipName = "lambda_package.zip"
if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
if (Test-Path $ZipName) { Remove-Item -Force $ZipName }
New-Item -ItemType Directory -Path $PackageDir | Out-Null
pip install psycopg2-binary pandas numpy requests boto3 --platform manylinux2014_x86_64 --target $PackageDir --implementation cp --python-version 3.12 --only-binary=:all: --upgrade
Copy-Item "lambda_function.py" -Destination "$PackageDir\lambda_function.py"
Start-Sleep -Seconds 5
Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipName -Force
$SizeMB = [math]::Round((Get-Item $ZipName).Length / 1MB, 1)
Write-Host "Done! $ZipName ($SizeMB MB)" -ForegroundColor Green
}