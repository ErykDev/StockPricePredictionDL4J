package org.pp.data;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.Objects;

public class CustomDataPrePreprocessor implements DataSetPreProcessor {

    @Getter
    double biggestNum;

    @Override
    public void preProcess(DataSet toPreProcess) {
        if (biggestNum == 0.0){
            double v1 = toPreProcess.getFeatures().maxNumber().doubleValue();
            double v2 = toPreProcess.getLabels().maxNumber().doubleValue();

            biggestNum = Math.max(v1, v2);
        }

        toPreProcess.setFeatures(toPreProcess.getFeatures().div(biggestNum));
        toPreProcess.setLabels(toPreProcess.getLabels().div(biggestNum));
    }

    public void revert(DataSet toRevert){
        toRevert.setFeatures(revert(toRevert.getFeatures()));
        toRevert.setLabels(revert(toRevert.getLabels()));
    }

    public INDArray revert(INDArray toRevert){
        return toRevert.mul(biggestNum);
    }
}
