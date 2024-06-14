import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor,BlipCaptionProcessor
from minigpt4.datasets.datasets.video_datasets import VideoChatGPTEvalDataset,VideoChatGPTEval_consistancy,Video_validation_Dataset,TVQAEVAL
parser = eval_parser()
parser.add_argument("--dataset", type=str, default='msvd', help="dataset to evaluate")
parser.add_argument("--add_subtitles",action='store_true',help="whether to add subtitles to the video")
parser.add_argument("--name", type=str, default='3_datasets', help="evaluation name")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start from video number")
parser.add_argument("--end", type=int, default=10000000, help="end at video number")
parser.add_argument("--need_speech",action='store_true',help="whether to speech")
parser.add_argument("--wav_base", type=str, default='', help="wav base path")
parser.add_argument("--result_path", type=str, default="/home/develop/fyy/MiniGPT4-video-main/ego-sentiment/result/non_sub_no_speech.json", help="wav base path")
parser.add_argument("--ann_path",type=str,default="/home/develop/fyy/MiniGPT4-video-main/ego-sentiment/ques2_test.json")
args = parser.parse_args()

print(args.ckpt)
print(args.name)
print(args.cfg_path)
if "test_configs/mistral_test_config.yaml" == args.cfg_path: 
    llm_name="mistral"
else:   
    llm_name="llama2"
print("using captions",args.add_subtitles)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION.copy()
conv_temp.system = ""
if args.dataset == 'video_chatgpt_generic':
    ann_path="datasets/evaluation_datasets/videochatgpt_benchmark/generic_qa.json"
    videos_path="/ibex/project/c2090/datasets/VideoInstruct100K/test_videos/Test_Videos"
    subtitles_path="/home/ataallka/minigpt_video/minigpt_multi_img/inference_subtitles"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/benchmark/generic"
    annotations_keys=['Q','A','video_name']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'video_chatgpt_temporal':
    ann_path="datasets/evaluation_datasets/videochatgpt_benchmark/temporal_qa.json"
    videos_path="/ibex/project/c2090/datasets/VideoInstruct100K/test_videos/Test_Videos"
    subtitles_path="/home/ataallka/minigpt_video/minigpt_multi_img/inference_subtitles"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/benchmark/temporal"
    annotations_keys=['Q','A','video_name']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'video_chatgpt_consistency':
    ann_path="datasets/evaluation_datasets/videochatgpt_benchmark/consistency_qa.json"
    videos_path="/ibex/project/c2090/datasets/VideoInstruct100K/test_videos/Test_Videos"
    subtitles_path="/home/ataallka/minigpt_video/minigpt_multi_img/inference_subtitles"
    annotations_keys=[['Q1','Q2'],'A','video_name']
    data = VideoChatGPTEval_consistancy(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys, add_subtitles=args.add_subtitles,llm_name=llm_name)
    
elif args.dataset == 'msrvtt':
    ann_path="datasets/evaluation_datasets/msrvtt/val_qa_edited.json"
    videos_path="/ibex/project/c2090/datasets/VideoInstruct100K/test_videos/MSRVTT/videos/all"
    subtitles_path="/home/ataallka/minigpt_video/minigpt_multi_img/inference_subtitles"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/msrvtt"
    annotations_keys=['question','answer','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=args.add_subtitles,llm_name=llm_name)

elif args.dataset == 'msvd':
    audio_conf_val = {'num_mel_bins': 128, 
                      'target_length': 1024, 
                      'freqm': 0,
                      'timem': 0,
                      'mixup': 0,
                      'mode':'val',
                      'mean':-4.2677393,
                      'std':4.5689974,
                      'noise':False,
                      'multilabel':False,
                      }  
    need_speech=args.need_speech
    wav_base=args.wav_base
    audio_conf=audio_conf_val
    # ann_path="/home/develop/fyy/MiniGPT4-video-main/msvd/datasets_evaluation_datasets_msvd_val_qa_edited.json"
    # videos_path="/home/develop/fyy/MiniGPT4-video-main/msvd/YouTubeClips"
    # subtitles_path="/home/develop/fyy/MiniGPT4-video-main/msvd/subtitles"
    # videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/msvd"
    # annotations_keys=['question','answer','video_id']
    ann_path="/home/develop/fyy/MiniGPT4-video-main/ego-sentiment/ques2_test.json"
    videos_path="/home/develop/fyy/video/subset/"
    subtitles_path="/home/develop/fyy/MiniGPT4-video-main/msvd/subtitles"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/msvd"
    annotations_keys=['q','a','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=args.add_subtitles,llm_name=llm_name,
                                    need_speech=need_speech,
            wav_base=wav_base,
            audio_conf=audio_conf_val)
elif args.dataset == 'ego-sentiment':
    audio_conf_val = {'num_mel_bins': 128, 
                      'target_length': 1024, 
                      'freqm': 0,
                      'timem': 0,
                      'mixup': 0,
                      'mode':'val',
                      'mean':-4.2677393,
                      'std':4.5689974,
                      'noise':False,
                      'multilabel':False,
                      }  
    need_speech=args.need_speech
    wav_base=args.wav_base
    audio_conf=audio_conf_val

    ann_path=args.ann_path
    videos_path="/mnt/sdc/develop/fyy/video/one_path"
    subtitles_path="/home/develop/fyy/vtt"
    videos_features_path=""
    annotations_keys=['q','a','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=args.add_subtitles,llm_name=llm_name,
                                    need_speech=need_speech,
            wav_base=wav_base,
            audio_conf=audio_conf_val, dataset=args.dataset)
elif args.dataset == 'activitynet':
    ann_path="datasets/evaluation_datasets/activityNet/test_qa.json"
    videos_path="/ibex/project/c2090/datasets/VideoInstruct100K/test_videos/Activity_net/Activity_net_videos"
    subtitles_path="/home/ataallka/minigpt_video/minigpt_multi_img/inference_subtitles/"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/activity_net"
    annotations_keys=['question','answer','video_id']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=args.add_subtitles,llm_name=llm_name)
elif args.dataset == 'tgif':
    ann_path="datasets/evaluation_datasets/tgif/Test_frameqa_question.json"
    videos_path="/ibex/project/c2090/datasets/VideoInstruct100K/test_videos/TGIF/mp4s"
    subtitles_path="/home/ataallka/minigpt_video/minigpt_multi_img/inference_subtitles"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/tgif"
    annotations_keys=['question','answer','gif_name']
    # annotations_keys=['question','description','gif_name']
    data = VideoChatGPTEvalDataset(vis_processor, videos_path, ann_path,subtitles_path,annotations_keys,videos_features_path, add_subtitles=False,llm_name=llm_name)
elif args.dataset == 'tvqa':
    # TVQA dataset
    ann_path="datasets/evaluation_datasets/tvqa_short/tvqa_val.json"
    videos_path= "/ibex/project/c2090/datasets/TVR_dataset/videos/video_files/frames_hq/"
    subtitles_path="/ibex/project/c2090/datasets/TVR_dataset/TVRetrieval/data/tvqa_preprocessed_subtitles.json"
    videos_features_path="/ibex/project/c2106/kirolos/videos_features/evaluation/tvqa"
    data = TVQAEVAL(vis_processor, videos_path, ann_path,subtitles_path,videos_features_path,add_subtitles=args.add_subtitles,llm_name=llm_name)

eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

minigpt4_predict = []
sub="subtitles" if args.add_subtitles else "no_subtitles"
if args.start == 0 and args.end == 10000000:
    save_path = f'results/{args.name}_{args.dataset}_{sub}.json'
else:
    print("start from video number",args.start)
    print("end at video number",args.end)
    save_path = f'results/{args.name}_{args.dataset}_{sub}_{args.start}_{args.end}.json'
save_path=args.result_path

os.makedirs("results", exist_ok=True)
c=0
pred_result = {}
gt_result = {}
if args.dataset == 'video_chatgpt_consistency':
    for images, texts_1,texts_2, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts_q1 = prepare_texts(texts_1, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            texts_q2 = prepare_texts(texts_2, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers_q1 = model.generate(images, texts_q1, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            models_answers_q2 = model.generate(images, texts_q2, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer_q1,model_answer_q2, gt_answer,text_q1,text_q2 in zip(videos_ids,models_answers_q1,models_answers_q2, gt_answers,texts_q1,texts_q2):
                result = dict()
                result['video_name'] = video_id
                result['Q1'] = text_q1.split('\n')[-1].replace('[/INST]','')
                result['Q2'] = text_q2.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred1'] = model_answer_q1
                result['pred2'] = model_answer_q2
                pred_result[video_id] = [model_answer_q1,model_answer_q2]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1

elif args.dataset == 'tvr':
    for images, texts, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                result['Q'] = text.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred'] = model_answer
                pred_result[video_id] = [model_answer]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1
elif args.dataset == 'ego_schema' or args.dataset == 'tvqa' or args.dataset == 'tvqa_long_videos':
    for images, texts, gt_answers, lengths,videos_ids in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.start<= c <args.end :
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                if args.dataset == 'tvqa_long_videos':
                    result['Q'] = text.split('\n\n')[1:]
                else:
                    result['Q'] = text.split('\n')[1:]
                result['A'] = gt_answer
                result['pred'] = model_answer
                pred_result[video_id] = [model_answer]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1
else:
    for images, texts, gt_answers, lengths,videos_ids, speech in tqdm(eval_dataloader,desc=f"Eval {args.dataset}"):
        if args.need_speech==False:
            speech=None
        if args.start<= c <args.end :
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=args.max_new_tokens, do_sample=False, lengths=lengths,num_beams=1, speech=speech)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                result['Q'] = text.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred'] = model_answer
                pred_result[video_id] = [model_answer]
                gt_result[video_id] = [gt_answer]
                minigpt4_predict.append(result)
            # save results every 100 videos to avoid losing results
            if c%100==0:
                with open(save_path, 'w') as f:
                    json.dump(minigpt4_predict, f)
        if c >= args.end :
            break
        c+=1

with open(save_path, 'w') as f:
    json.dump(minigpt4_predict, f)
print("saved results to",save_path)
# save results
# bleu_save_path = f'results/{args.name}_{args.dataset}_bleu.json'
# cider_save_path = f'results/{args.name}_{args.dataset}_cider.json'
# chatgpt_eval_save_path = f'results/{args.name}_{args.dataset}_chatgpt_eval.json'
# bleu_results=eval_bleu(minigpt4_predict)
# with open(bleu_save_path, 'w') as f:
#     json.dump(bleu_results, f)
# print("bleu_results",bleu_results)
# cider_results=eval_cider(pred_result,gt_result)
# with open(cider_save_path, 'w') as f:
#     json.dump(cider_results, f)
# print("mean_cider_scores:",cider_results['mean_cider_scores'])

# chatgpt_results=chat_gpt_eval(pred_result,gt_result)

# with open(chatgpt_eval_save_path, 'w') as f:
#     json.dump(chatgpt_results, f)
# print("avg_chatgpt_score",chatgpt_results['avg_chatgpt_score'])
# print(chatgpt_results)


