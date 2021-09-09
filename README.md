## StockPricesPredictionDL4J
Lstm based stock price prediction

![Example result](https://github.com/BadlyDrunkScotsman/StockPricesPredictionDL4J/blob/main/Test.png)

### How to run
* Import project dependencies with Gradle
* If you own an NVIDIA card with a CUDA support or have cpu that supports avx2 you can enable the usage in Gradle file

Model trained on NSE-TATAGLOBAL is avilable [here](https://drive.google.com/file/d/1hZvteE_rXenfwk6t4yNBfTbajlNdoAm9/view?usp=sharing)

### Training
* The project is using a csv data-format. See [example](https://github.com/BadlyDrunkScotsman/StockPricesPredictionDL4J/blob/main/NSE-TATAGLOBAL.csv)
* If you wish to use your own data simply paste it into the project folder in csv format if you have a custom csv edit StockCSVDataSetFetcher.java

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.




