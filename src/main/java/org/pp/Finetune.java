package org.pp;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.pp.component.UIServerComponent;
import org.pp.data.CustomDataPrePreprocessor;
import org.pp.data.StockCSVDataSetFetcher;
import org.pp.nn.NeuralNetwork;
import java.io.File;

@Slf4j
public class Finetune {
    @SneakyThrows
    public static void main(String[] args) {
        CustomDataPrePreprocessor normalizer = new CustomDataPrePreprocessor();

        UIServerComponent uiServerComponent = new UIServerComponent();

        File csvFile = new File("NSE-TATAGLOBAL.csv");

        int inpNum = 50;
        int outNum = 1;

        MultiLayerNetwork network = NeuralNetwork.getNetModel(inpNum, outNum);
        network.init();

        StockCSVDataSetFetcher dataSetFetcher = new StockCSVDataSetFetcher(csvFile, inpNum, outNum);
        BaseDatasetIterator datasetIterator = new BaseDatasetIterator(4, dataSetFetcher.totalExamples(), new StockCSVDataSetFetcher(csvFile, inpNum, outNum));
        normalizer.fit(datasetIterator);
        datasetIterator.setPreProcessor(normalizer);

        uiServerComponent.reinitialize(network);

        log.info(network.summary());

        int epochNum = 250;

        for (int i = 0; i < epochNum; i++) {
            network.fit(datasetIterator);

            log.info(String.format("Epoch: %s", i));

            network.rnnClearPreviousState();
            System.gc();
        }

        network.save(new File("network.zip"));
    }
}
