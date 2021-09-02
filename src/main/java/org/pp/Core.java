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
public class Core {
    @SneakyThrows
    public static void main(String[] args) {
        CustomDataPrePreprocessor normalizer = new CustomDataPrePreprocessor();

        UIServerComponent uiServerComponent = new UIServerComponent();

        File csvFile = new File("A_data.csv");

        int inpNum = 50;
        int outNum = 1;

        MultiLayerNetwork network = NeuralNetwork.getNetModel(inpNum, outNum);
        network.init();

        BaseDatasetIterator datasetIterator = new BaseDatasetIterator(32,64, new StockCSVDataSetFetcher(csvFile, inpNum, outNum));
        datasetIterator.setPreProcessor(normalizer);

        uiServerComponent.reinitialize(network);

        int epochNum = 3000;

        for (int i = 0; i < epochNum; i++) {
            double lr = calcLearningRate(i);

            network.setLearningRate(lr);
            network.fit(datasetIterator);

            System.gc();
        }
        network.save(new File("network.zip"));
    }

    private static double calcLearningRate(int epochNum){
        return epochNum <= 2000 ? 1e-3 : 1e-4;
    }
}
