package org.pp;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.pp.data.CustomDataPrePreprocessor;
import org.pp.data.StockCSVDataSetFetcher;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Collectors;

@Slf4j
public class Predict {
    static int inpNum = 50;
    static int outNum = 25;

    static CustomDataPrePreprocessor normalizer = new CustomDataPrePreprocessor();

    @SneakyThrows
    public static void main(String[] args) {
        File csvFile = new File("NSE-TATAGLOBAL.csv"); //data for normalization scale
        StockCSVDataSetFetcher dataSetFetcher = new StockCSVDataSetFetcher(csvFile, inpNum, outNum);

        BaseDatasetIterator datasetIterator = new BaseDatasetIterator(1, dataSetFetcher.totalExamples(), new StockCSVDataSetFetcher(csvFile, inpNum, outNum));
        normalizer.fit(datasetIterator);


        MultiLayerNetwork network = MultiLayerNetwork.load(new File("network.zip"), true);
        network.init();

        double[] data = readData();

        INDArray features = Nd4j.create(data, 1, data.length, 1);

        features = normalizer.preProcess(features);

        INDArray output = predictSteps(network, features, outNum);

        output = normalizer.revert(output);

        final XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(generateExpectedXYSeries(data));
        dataset.addSeries(generateOutputXYSeries(output));


        JFreeChart xylineChart = ChartFactory.createXYLineChart(
                "NSE-TATAGLOBAL",
                "Days",
                "Price",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);


        int width = 640;   /* Width of the image */
        int height = 480;  /* Height of the image */
        File XYChart = new File( "Predict.png" );
        ChartUtilities.saveChartAsPNG(XYChart, xylineChart, width, height);
    }

    @SneakyThrows
    private static double[] readData(){
        double[] data = new double[inpNum];

        ArrayList<String> allLines = Files.lines(Paths.get("NSE-TATAGLOBAL.csv"))
                .skip(1) // skipping csv headers
                .collect(Collectors.toCollection(ArrayList::new));

        for (int i = 0; i < inpNum; i++)
            data[i] = Double.parseDouble(allLines.get(i).split(",")[5]);

        return data;
    }

    private static INDArray predictSteps(MultiLayerNetwork network, INDArray input, int steps){
        INDArray tempInput = input.dup();
        INDArray stepsValues = Nd4j.create(1, steps);

        double outputNorm = 0.0;

        for (int i = 0; i < steps; i++) {
            if (i == 0)
                //calc outputNorm value
                outputNorm = calcOutputNorm(network, tempInput);

            double output = network.output(tempInput).getDouble(0, 0) - outputNorm;

            stepsValues.putScalar(0, i, output);

            //moving array by 1 pos
            for (int j = 0; j < inpNum-1; j++)
                tempInput.putScalar(0, j,0, tempInput.getDouble(0, j+1, 0));

            //adding predicted output
            tempInput.putScalar(0,inpNum-1,0, output);
        }
        return stepsValues;
    }

    private static double calcOutputNorm(MultiLayerNetwork network, INDArray input){
        return Math.round(Math.abs(network.output(input).getDouble(0, 0) - input.getDouble(0, inpNum - 1, 0)) * 100.0) / 100.0;
    }

    private static XYSeries generateExpectedXYSeries(double[] data){
        XYSeries expectedSeries = new XYSeries("Input");

        for (int i = 0; i < inpNum; i++)
            expectedSeries.add(i, data[i]);

        return expectedSeries;
    }

    private static XYSeries generateOutputXYSeries(INDArray expOutput){
        XYSeries expectedSeries = new XYSeries("Predicted");

        for (int i = 0; i < outNum; i++)
            expectedSeries.add(i+(inpNum-1), expOutput.getDouble(0, i));

        return expectedSeries;
    }
}
