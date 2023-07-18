from typing import Tuple

import torch

TOTAL_ROW_ID = -1
TP_COL_ID = 0
GP_COL_ID = 1
PP_COL_ID = 2
TOTAL_SAMPLE_ID = 1

class F1Score:
    """
    A Call Class for f1 calculation in Pytorch.
    
    Args:
        average (str): 'macro', 'micro', 'weighted'. Types of F1 score.
    """


    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')
    
    @staticmethod
    def calc_accumulate_components(acc_mat: torch.Tensor, predict: torch.Tensor, target: torch.Tensor):
        """
        Caculate accumulate TP, FP, FN for each target class ID, and total

        Args:
            acc_mat (Tensor): size (C+1) x 3, C rows of class ID, last row is total sample row
                Col is value of: TP (True Positive Samples), GP (Ground Truth Poistive Samples), PP (Predicted Positive)
                Except last row: store total true positive samples, total samples, total samples

        Return:
            Updated acc_mat
        """

        for id in torch.cat((predict, target)).unique():
            TP = torch.logical_and(torch.eq(target, predict), torch.eq(target, id)).sum().float()
            GP = torch.eq(target, id).sum().float()
            PP = torch.eq(predict, id).sum().float()
            acc_mat[id] += torch.tensor([TP, GP, PP])
            acc_mat[TOTAL_ROW_ID] += torch.tensor([TP, GP, GP])
        
        return acc_mat

    @staticmethod
    def calc_f1_micro(acc_mat: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            acc_mat: tensor accumulated matrix

        Returns:
            f1_score
        """
        TP = acc_mat[TOTAL_ROW_ID, TP_COL_ID].float()
        f1_score = torch.div(TP, acc_mat[TOTAL_ROW_ID, TOTAL_SAMPLE_ID]) # f1_score = TP/total_sample (total_PP = total_GP = total_sample)
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(acc_mat: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            acc_mat (Tensor): tensor accumulated matrix
            label_id (int): id of current label

        Returns:
            tupple: (f1 score, ground positive samples for label)
        """
        TP = acc_mat[label_id, TP_COL_ID].float()
        GP = acc_mat[label_id, GP_COL_ID].float()
        PP = acc_mat[label_id, PP_COL_ID].float()
        # precision for label
        precision = torch.div(TP, PP)

        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(TP),
                                precision)

        # recall for label
        recall = torch.div(TP, GP)
        
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(TP), f1)
        return f1, GP
    
    @staticmethod
    def calc_accuracy_count_for_label(acc_mat: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            acc_mat (Tensor): tensor accumulated matrix
            label_id (int): id of current label

        Returns:
            tupple: (f1 score, ground positive samples for label)
        """
        TP = acc_mat[label_id, TP_COL_ID].float()
        GP = acc_mat[label_id, GP_COL_ID].float()
        PP = acc_mat[label_id, PP_COL_ID].float()

        accuracy = torch.div(TP, GP)
        
        accuracy = torch.where(torch.isnan(accuracy),
                                torch.zeros_like(accuracy).type_as(TP),
                                accuracy)
        return TP, GP

    def __call__(self, acc_mat: torch.Tensor, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            acc_mat: tensor ((C+1) x 3)) store accumulated TP, GP (ground), PP (predict)
            predict: 2D tensor (size B x C) 
            target: 1D tensor (size C) value range [0, C-1]

        Returns:
            tupple: (f1 score, acc_mat)
        """
        predict = predict.argmax(dim=1)
        acc_mat = self.calc_accumulate_components(acc_mat, predict, target)

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(acc_mat), acc_mat

        f1_score = 0
        for label_id in torch.cat((predict,target)).unique():
            f1, GP = self.calc_f1_count_for_label(acc_mat, label_id)

            if self.average == 'weighted':
                f1_score += f1 * GP
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, acc_mat[TOTAL_ROW_ID, TOTAL_SAMPLE_ID])
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, acc_mat.shape[0] - 1)

        return f1_score, acc_mat
