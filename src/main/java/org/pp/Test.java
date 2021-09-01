package org.pp;

import lombok.SneakyThrows;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.pp.data.StockCSVDataSetFetcher;
import org.pp.nn.NeuralNetwork;

import java.awt.*;
import java.io.File;

public class Test {

    static int inpNum = 100;
    static int outNum = 10;

    @SneakyThrows
    public static void main(String[] args) {
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(-1,1);

        File csvFile = new File("A_data.csv");



        MultiLayerNetwork network = MultiLayerNetwork.load(new File("network.zip"), true);
        network.init();

        BaseDatasetIterator datasetIterator = new BaseDatasetIterator(1,10, new StockCSVDataSetFetcher(csvFile, inpNum, outNum));
        normalizer.fit(datasetIterator);
        datasetIterator.setPreProcessor(normalizer);


        DataSet dataSet = datasetIterator.next();

        INDArray output = network.output(dataSet.getFeatures());

        normalizer.revert(dataSet);
        normalizer.revertLabels(output);



        final XYSeriesCollection dataset = new XYSeriesCollection( );
        dataset.addSeries(generateExpectedXYSeries(dataSet));
        dataset.addSeries(generateOutputXYSeries(output));


        JFreeChart xylineChart = ChartFactory.createXYLineChart(
                "Predictions" ,
                "X" ,
                "Price" ,
                dataset ,
                PlotOrientation.VERTICAL ,
                true , true , false);


        int width = 640;   /* Width of the image */
        int height = 480;  /* Height of the image */
        File XYChart = new File( "Results.jpeg" );
        ChartUtilities.saveChartAsJPEG( XYChart, xylineChart, width, height);
    }

    private static XYSeries generateExpectedXYSeries(DataSet dataSet){
        XYSeries expectedSeries = new XYSeries( "Expected" );

        INDArray expInput = dataSet.getFeatures();
        INDArray expOutput= dataSet.getLabels();

        for (int i = 0; i < inpNum; i++)
            expectedSeries.add(i, expInput.getDouble(0,i,0));

        for (int i = 0; i < outNum; i++)
            expectedSeries.add(i+inpNum, expOutput.getDouble(0,i));

        return expectedSeries;
    }

    private static XYSeries generateOutputXYSeries(INDArray expOutput){
        XYSeries expectedSeries = new XYSeries( "Predicted" );

        for (int i = 0; i < outNum; i++)
            expectedSeries.add(i+inpNum, expOutput.getDouble(0,i));

        return expectedSeries;
    }

}
