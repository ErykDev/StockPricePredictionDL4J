package org.pp.data;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

public class CustomDataPrePreprocessor implements DataSetPreProcessor {

    @Getter
    double biggestNum = 0.0;

    @Getter
    double smallestNum = 0.0;

    @Override
    public void preProcess(DataSet toPreProcess) {
        toPreProcess.setFeatures(preProcess(toPreProcess.getFeatures()));
        toPreProcess.setLabels(preProcess(toPreProcess.getLabels()));
    }

    public INDArray preProcess(INDArray toPreProcess) {
        return toPreProcess.sub(smallestNum).div(biggestNum - smallestNum).mul(2.0).sub(1.0);
    }

    public void revert(DataSet toRevert){
        toRevert.setFeatures(revert(toRevert.getFeatures()));
        toRevert.setLabels(revert(toRevert.getLabels()));
    }

    public INDArray revert(INDArray toRevert){
        return toRevert.add(1.0).div(2.0).add(smallestNum).mul(biggestNum - smallestNum);
    }

    public void fit(BaseDatasetIterator iterator){
        while (iterator.hasNext()) {
            DataSet dataSet = iterator.next();

            double b1 = dataSet.getFeatures().maxNumber().doubleValue();
            double b2 = dataSet.getLabels().maxNumber().doubleValue();
            double tempBiggestNum = Math.max(b1, b2);

            if (tempBiggestNum > biggestNum)
                this.biggestNum = tempBiggestNum;

            double s1 = dataSet.getFeatures().minNumber().doubleValue();
            double s2 = dataSet.getLabels().minNumber().doubleValue();
            double tempSmallestNum = Math.min(s1, s2);

            if (tempSmallestNum < smallestNum)
                this.smallestNum = tempSmallestNum;
        }

        iterator.reset();
    }
}
