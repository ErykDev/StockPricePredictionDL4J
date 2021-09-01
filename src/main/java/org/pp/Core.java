package org.pp;


import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.pp.component.UIServerComponent;
import org.pp.data.StockCSVDataSetFetcher;
import org.pp.nn.NeuralNetwork;
import java.io.File;


@Slf4j
public class Core {
    @SneakyThrows
    public static void main(String[] args) {
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0.0, 1.0);

        UIServerComponent uiServerComponent = new UIServerComponent();

        File csvFile = new File("A_data.csv");

        int inpNum = 100;
        int outNum = 10;

        MultiLayerNetwork network = NeuralNetwork.getNetModel(inpNum, outNum);
        network.init();

        BaseDatasetIterator datasetIterator = new BaseDatasetIterator(32,256, new StockCSVDataSetFetcher(csvFile, inpNum, outNum));
        normalizer.fit(datasetIterator);
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
