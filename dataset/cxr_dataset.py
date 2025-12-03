import os
import cv2
import pandas as pd
import numpy as np
import jpeg4py as jpeg
from torch import from_numpy
from torch.utils.data import Dataset


class CxrDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if all([c in self.df.columns for c in self.cfg['classes']]):
            label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        path = self.df.iloc[index]["path"]
        path = os.path.join(self.cfg['data_dir'], path)
        resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

        if os.path.exists(resized_path):
            img = jpeg.JPEG(resized_path).decode()
            if os.path.exists(path):
                os.remove(path)
            assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
        else:
            img = jpeg.JPEG(path).decode()
            img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(resized_path, img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label 


class CxrBalancedDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        class_name = self.cfg['classes'][index%len(self.cfg['classes'])]
        df = self.df[self.df[class_name] == 1].sample(1).iloc[0]

        label = df[self.cfg['classes']].to_numpy().astype(np.float32)    

        path = df["path"]
        path = os.path.join(self.cfg['data_dir'], path)
        resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

        if os.path.exists(resized_path):
            img = jpeg.JPEG(resized_path).decode()
            assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
        else:
            img = jpeg.JPEG(path).decode()
            img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(resized_path, img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label


class CxrStudyIdDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df = self.df.get_group(self.study_ids[index])
        if len(df) > 4:
            df = df.sample(4)

        if all([c in df.columns for c in self.cfg['classes']]):
            label = df[self.cfg['classes']].iloc[0].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        imgs = []
        for i in range(len(df)):
            path = df.iloc[i]["path"]
            path = os.path.join(self.cfg['data_dir'], path)
            resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")
            if os.path.exists(resized_path):
                img = jpeg.JPEG(resized_path).decode()            
                if os.path.exists(path):
                    os.remove(path)
                assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
            else:
                img = jpeg.JPEG(path).decode()
                img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(resized_path, img)

            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']   
                img = np.moveaxis(img, -1, 0)
            imgs.append(img)

        img = np.stack(imgs, axis=0)    
        img = np.concatenate([img, np.zeros((4-len(df), 3, self.cfg['size'], self.cfg['size']))], axis=0).astype(np.float32)
        return img, label


class CxrStudyIdDataset2(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self.transform = transform

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        df_group = self.df.get_group(study_id).copy()
        
        # 1. 뷰 그룹핑 (대소문자 무시)
        df_group['ViewUpper'] = df_group['ViewPosition'].astype(str).str.upper()
        
        pa_df = df_group[df_group['ViewUpper'].isin(['PA', 'AP'])]
        lat_df = df_group[df_group['ViewUpper'].isin(['LATERAL', 'LL', 'RL'])]
        
        # 나머지(Others)는 필요시 사용 (여기선 PA/LAT 우선 채우기 위해 보류)
        others_df = df_group[~df_group.index.isin(pa_df.index) & ~df_group.index.isin(lat_df.index)]

        # 2. 슬롯 채우기 (목표: [PA, PA, LAT, LAT])
        # 학습 시에는 랜덤 샘플링(Augmentation 효과), 평가 시에는 고정
        
        def select_imgs(sub_df, count):
            if len(sub_df) == 0: return []
            if len(sub_df) >= count:
                # 학습이면 랜덤, 평가면 앞부분
                return [sub_df.sample(count)]
            else:
                return [sub_df]

        final_rows = []
        
        # Slot 0, 1: PA (최대 2장)
        final_rows.extend(select_imgs(pa_df, 2))
        
        # Slot 2, 3: LAT (최대 2장)
        # 만약 PA가 2장 꽉 찼으면 LAT는 뒤에 붙지만, 
        # 리스트 구조상 PA들을 먼저 append 했으므로 나중에 concat하면 자연스럽게 앞쪽으로 옴.
        # 하지만 우리는 [PA구역, LAT구역]을 고정하고 싶으므로 별도 리스트로 관리.
        
        rows_pa = select_imgs(pa_df, 2)
        rows_lat = select_imgs(lat_df, 2)
        
        # DataFrame으로 변환
        d_pa = pd.concat(rows_pa) if rows_pa else pd.DataFrame()
        d_lat = pd.concat(rows_lat) if rows_lat else pd.DataFrame()
        
        # 이미지 로드 함수
        def load_images(target_df):
            imgs = []
            for i in range(len(target_df)):
                path = target_df.iloc[i]["path"]
                path = os.path.join(self.cfg['data_dir'], path)
                resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")
                
                # 로드 로직 (JPEG -> CV2 fallback)
                if os.path.exists(resized_path):
                    try:
                        with open(resized_path, 'rb') as f: img = jpeg.decode_jpeg(f.read())
                    except:
                        img = cv2.imread(resized_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(path)
                    if img is None: img = np.zeros((self.cfg['size'], self.cfg['size'], 3), dtype=np.uint8)
                    else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)

                if self.transform:
                    transformed = self.transform(image=img)
                    img = transformed['image']   
                    img = np.moveaxis(img, -1, 0)
                imgs.append(img)
            return imgs

        imgs_pa = load_images(d_pa)
        imgs_lat = load_images(d_lat)

        # 3. 텐서 만들기 (고정된 자리 배치)
        # [Slot 0, 1] : PA
        t_pa = np.stack(imgs_pa, axis=0) if imgs_pa else np.zeros((0, 3, self.cfg['size'], self.cfg['size']))
        if len(imgs_pa) < 2: # 패딩
            t_pa = np.concatenate([t_pa, np.zeros((2-len(imgs_pa), 3, self.cfg['size'], self.cfg['size']))], axis=0)

        # [Slot 2, 3] : LAT
        t_lat = np.stack(imgs_lat, axis=0) if imgs_lat else np.zeros((0, 3, self.cfg['size'], self.cfg['size']))
        if len(imgs_lat) < 2: # 패딩
            t_lat = np.concatenate([t_lat, np.zeros((2-len(imgs_lat), 3, self.cfg['size'], self.cfg['size']))], axis=0)

        # 합체: [4, 3, H, W]
        final_img = np.concatenate([t_pa, t_lat], axis=0).astype(np.float32)

        # 4. Label
        if all([c in df_group.columns for c in self.cfg['classes']]):
            label = df_group[self.cfg['classes']].iloc[0].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        return final_img, label