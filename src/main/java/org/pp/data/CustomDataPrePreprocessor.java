package org.pp.data;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

public class CustomDataPrePreprocessor implements DataSetPreProcessor {

    @Getter
    double biggestNum;

    @Override
    public void preProcess(DataSet toPreProcess) {
        toPreProcess.setFeatures(preProcess(toPreProcess.getFeatures()));
        toPreProcess.setLabels(preProcess(toPreProcess.getLabels()));
    }

    public void fit(BaseDatasetIterator iter){
        while (iter.hasNext()) {
            DataSet dataSet = iter.next();

            double v1 = dataSet.getFeatures().maxNumber().doubleValue();
            double v2 = dataSet.getLabels().maxNumber().doubleValue();
            double tempBiggestNum = Math.max(v1, v2);

            if (tempBiggestNum > biggestNum)
                this.biggestNum = tempBiggestNum;
        }
    }

    public INDArray preProcess(INDArray toPreProcess) {
        return toPreProcess.div(biggestNum);
    }

    public void revert(DataSet toRevert){
        toRevert.setFeatures(revert(toRevert.getFeatures()));
        toRevert.setLabels(revert(toRevert.getLabels()));
    }

    public INDArray revert(INDArray toRevert){
        return toRevert.mul(biggestNum);
    }
}
