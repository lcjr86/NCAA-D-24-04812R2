# Create a mosaic

from PIL import Image
import cv2
import os
import numpy as np

def generate_mosaic(full_path_img1, full_path_img2, full_path_img3, full_path_img4, full_path_img5, full_path_img6, full_path_img7, full_path_output):

    # Step 1: Load the images
    image1 = Image.open(full_path_img1)
    image2 = Image.open(full_path_img2)
    image3 = Image.open(full_path_img3)
    image4 = Image.open(full_path_img4)
    image5 = Image.open(full_path_img5)
    image6 = Image.open(full_path_img6)
    image7 = Image.open(full_path_img7)

    # # Step 2: Resize the images (if needed)
    # base_size = (200, 200)
    # image1 = image1.resize(base_size)
    # image2 = image2.resize(base_size)
    # image3 = image3.resize(base_size)
    # image4 = image4.resize(base_size)

    # Step 3: Create a new image object for the canvas
    canvas = Image.new('RGB', (1500, 1500), 'white')

    # Step 4: Paste the images
    canvas.paste(image1, (0, 0))
    canvas.paste(image2, (500, 0))
    
    canvas.paste(image3, (0, 500))
    canvas.paste(image4, (500, 500))
    
    canvas.paste(image5, (0, 1000))
    canvas.paste(image6, (500, 1000))
    canvas.paste(image7, (1000, 1000))

    # Step 5: Save the new image
    canvas.save(full_path_output)

def generate_mosaic_and_features(full_path_img1, full_path_img2, full_path_output):

    # Step 1: Load the images
    image1 = Image.open(full_path_img1)
    image2 = Image.open(full_path_img2)

    # # Step 2: Resize the images (if needed)
    # base_size = (200, 200)
    # image1 = image1.resize(base_size)
    # image2 = image2.resize(base_size)
    # image3 = image3.resize(base_size)
    # image4 = image4.resize(base_size)

    # Step 3: Create a new image object for the canvas
    canvas = Image.new('RGB', (2500, 1500), 'white')

    # Step 4: Paste the images
    canvas.paste(image1, (0, 0))
    canvas.paste(image2, (1500, 0))

    # Step 5: Save the new image
    canvas.save(full_path_output)


def create_mosaic_video(mosaic_path, output_path, video_name, frame_time=1):

    # Sort and filter images
    images = [img for img in os.listdir(mosaic_path) if img.endswith(".png")]
    images.sort()

    # Read the first image to obtain dimensions
    sample = cv2.imread(os.path.join(mosaic_path, images[0]))

    # Define video properties
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    height, width, layers = sample.shape
    output_filepath = os.path.join(output_path, video_name)
    video = cv2.VideoWriter(output_filepath, fourcc, 1, (width, height))

    # Write video
    for image in images:
        img_path = os.path.join(mosaic_path, image)
        frame = cv2.imread(img_path)

        # Advanced frame manipulations could be inserted here if necessary

        for _ in range(frame_time):
            video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def generate_numerical_filename(index, total_images):
    length = len(str(total_images))
    return f"{str(index).zfill(length)}"


# Add technical indicators method to the data frame

def add_technical_indicators(df):

    data = df.copy()
    # Moving Average
    data['MA_5'] = data['close'].rolling(5).mean()
    data['MA_9'] = data['close'].rolling(9).mean()
    data['MA_10'] = data['close'].rolling(10).mean()
    data['MA_20'] = data['close'].rolling(20).mean()
    data['MA_26'] = data['close'].rolling(26).mean()
    data['MA_50'] = data['close'].rolling(50).mean()
    data['MA_52'] = data['close'].rolling(52).mean()
    data['MA_100'] = data['close'].rolling(100).mean()
    data['MA_200'] = data['close'].rolling(200).mean()

    # Exponential Moving Average
    data['EMA_3'] = data['close'].ewm(span=3, adjust=False).mean()
    data['EMA_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA_9'] = data['close'].ewm(span=9, adjust=False).mean()
    data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['EMA_53'] = data['close'].ewm(span=53, adjust=False).mean()
    data['EMA_100'] = data['close'].ewm(span=100, adjust=False).mean()
    data['EMA_200'] = data['close'].ewm(span=200, adjust=False).mean()

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

    # Bollinger Bands
    data['BB_up'] = data['MA_20'] + 2 * data['close'].rolling(20).std()
    data['BB_down'] = data['MA_20'] - 2 * data['close'].rolling(20).std()

    # RSI
    delta = data['close'].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['RSI'] = 100 - (100 / (1 + (up / down).rolling(14).mean().abs()))

    # Stochastic Oscillator
    data['SO_k'] = 100 * ((data['close'] -  data['low'].rolling(14).min()) / (data['high'].rolling(14).max() - data['low'].rolling(14).min()))
    data['SO_d'] = data['SO_k'].rolling(3).mean()

    # Williams %R
    data['Williams_%R'] = -100 * ((data['high'].rolling(14).max() - data['close']) / (data['high'].rolling(14).max() - data['low'].rolling(14).min()))

    # ATR
    data['TR'] = data['high'] - data['low']
    data['ATR'] = data['TR'].rolling(14).mean()

    # ADX
    data['DM_plus'] = data['high'].diff(1)
    data['DM_minus'] = data['low'].diff(1)
    data['DM_plus'][data['DM_plus'] < 0] = 0
    data['DM_minus'][data['DM_minus'] > 0] = 0
    data['DM_plus'][data['DM_plus'] < data['DM_minus']] = 0
    data['DM_minus'][data['DM_minus'] < data['DM_plus']] = 0
    data['DM_plus'] = data['DM_plus'].abs()
    data['DM_minus'] = data['DM_minus'].abs()
    data['DM_plus'][data['DM_plus'] < data['DM_minus']] = 0
    data['DM_minus'][data['DM_minus'] < data['DM_plus']] = 0
    data['DM_plus'][data['DM_plus'] == data['DM_minus']] = 0
    data['DM_minus'][data['DM_plus'] == data['DM_minus']] = 0
    data['DI_plus'] = 100 * (data['DM_plus'].rolling(14).mean() / data['ATR'])
    data['DI_minus'] = 100 * (data['DM_minus'].rolling(14).mean() / data['ATR'])
    data['DX'] = 100 * (data['DI_plus'] - data['DI_minus']) / (data['DI_plus'] + data['DI_minus'])
    data['ADX'] = data['DX'].rolling(14).mean()
    data['ADX'] = data['ADX'].fillna(0)

    # CCI
    data['CCI'] = (data['close'] - data['MA_20']) / (0.015 * data['close'].rolling(20).std())

    # Force Index
    data['FI'] = data['close'].diff(1) * data['volume']

    # EOM
    data['EOM'] = data['volume'] / data['volume'].rolling(14).mean()
    data['EOM_MA'] = data['EOM'].rolling(14).mean()

    # Vortex
    data['VM_plus'] = data['high'].diff(1).abs() + data['low'].diff(1).abs()
    data['VM_minus'] = data['high'].diff(1).abs() + data['low'].diff(1).abs()
    data['VM_plus'][data['VM_plus'] < data['VM_minus']] = 0
    data['VM_minus'][data['VM_minus'] < data['VM_plus']] = 0
    data['VM_plus'] = data['VM_plus'].rolling(14).sum() / data['TR'].rolling(14).sum()
    data['VM_minus'] = data['VM_minus'].rolling(14).sum() / data['TR'].rolling(14).sum()
    data['VI_plus'] = 100 * data['VM_plus'].rolling(14).mean()
    data['VI_minus'] = 100 * data['VM_minus'].rolling(14).mean()
    data['Vortex'] = data['VI_plus'] - data['VI_minus']
    
    # KST Oscillator
    RCMA_10 = data['close'].diff(10) / data['close'].rolling(10).mean()
    RCMA_10 = RCMA_10.fillna(0)
    RCMA_15 = data['close'].diff(15) / data['close'].rolling(15).mean()
    RCMA_15 = RCMA_15.fillna(0)
    RCMA_20 = data['close'].diff(20) / data['close'].rolling(20).mean()
    RCMA_20 = RCMA_20.fillna(0)
    RCMA_30 = data['close'].diff(30) / data['close'].rolling(30).mean()
    RCMA_30 = RCMA_30.fillna(0)
    data['KST'] = 100 * (RCMA_10 + 2 * RCMA_15 + 3 * RCMA_20 + 4 * RCMA_30)
    data['KST_signal'] = data['KST'].rolling(9).mean()

    # Ichimoku Cloud
    data['IC_line'] = (data['MA_9'] + data['MA_26']) / 2
    data['IC_base'] = (data['MA_52'] + data['MA_26']) / 2
    data['IC_lead_1'] = (data['IC_line'] + data['IC_base']) / 2
    data['IC_lead_2'] = (data['MA_52'] + data['MA_52']) / 2
    data['IC_lead_1'] = data['IC_lead_1'].shift(26)
    data['IC_lead_2'] = data['IC_lead_2'].shift(26)

    # Aroon
    data['Aroon_up'] = 100 * data['high'].rolling(25).apply(lambda x: x.argmax(), raw=True) / 25
    data['Aroon_down'] = 100 * data['low'].rolling(25).apply(lambda x: x.argmin(), raw=True) / 25
    data['Aroon'] = data['Aroon_up'] - data['Aroon_down']

    # PSAR
    # data['PSAR'] = data['close'].copy()
    # data['PSAR_up'] = True
    # data['PSAR_down'] = False
    # data['PSAR_acc'] = 0.02
    # data['PSAR_acc'][data['PSAR_acc'] > 0.2] = 0.2
    # data['PSAR_acc'][data['PSAR_acc'] < 0.02] = 0.02
    # data['PSAR_acc_up'] = data['PSAR_acc'].copy()
    # data['PSAR_acc_down'] = data['PSAR_acc'].copy()
    # data['PSAR_ep'] = data['low'].copy()
    # data['PSAR_ep'][data['PSAR_up']] = data['high'] - data['PSAR_acc_up'] * (data['high'] - data['PSAR_ep'])
    # data['PSAR_ep'][data['PSAR_down']] = data['low'] + data['PSAR_acc_down'] * (data['PSAR_ep'] - data['low'])
    # data['PSAR'][data['PSAR_up']] = data['PSAR'][data['PSAR_up']].shift(1)
    # data['PSAR'][data['PSAR_down']] = data['PSAR'][data['PSAR_down']].shift(1)
    # data['PSAR'][data['PSAR_up']] = np.fmin(data['PSAR_ep'], data['PSAR'][data['PSAR_up']])
    # data['PSAR'][data['PSAR_down']] = np.fmax(data['PSAR_ep'], data['PSAR'][data['PSAR_down']])
    # data['PSAR_up'] = data['PSAR_up'].shift(1)
    # data['PSAR_down'] = data['PSAR_down'].shift(1)
    # data['PSAR_up'][data['PSAR_up']] = data['low'] > data['PSAR'][data['PSAR_up']]
    # data['PSAR_down'][data['PSAR_down']] = data['high'] < data['PSAR'][data['PSAR_down']]
    # data['PSAR_up'][data['PSAR_up']] = data['PSAR_down']
    # data['PSAR_down'][data['PSAR_down']] = data['PSAR_up']
    # data['PSAR_up'][data['PSAR_up']] = data['high']
    # data['PSAR_down'][data['PSAR_down']] = data['low']
    # data['PSAR_acc_up'][data['PSAR_up']] = data['PSAR_acc_up'] + 0.02
    # data['PSAR_acc_down'][data['PSAR_down']] = data['PSAR_acc_down'] + 0.02
    # data['PSAR_acc_up'][data['PSAR_acc_up'] > 0.2] = 0.2
    # data['PSAR_acc_down'][data['PSAR_acc_down'] > 0.2] = 0.2
    # data['PSAR_acc_up'][data['PSAR_down']] = 0.02
    # data['PSAR_acc_down'][data['PSAR_up']] = 0.02
    # data['PSAR_acc'] = data['PSAR_acc_up'].copy()
    # data['PSAR_acc'][data['PSAR_up']] = data['PSAR_acc_up']
    # data['PSAR_acc'][data['PSAR_down']] = data['PSAR_acc_down']
    # data['PSAR_up'][data['PSAR_up']] = True
    # data['PSAR_down'][data['PSAR_down']] = False

    # TSI
    data['TSI'] = 100 * (data['close'].diff(1).ewm(span=25, adjust=False).mean() / data['close'].diff(1).abs().ewm(span=25, adjust=False).mean())
    data['TSI_signal'] = data['TSI'].ewm(span=13, adjust=False).mean()

    # Ultimate Oscillator
    data['BP'] = data['close'] - np.fmin(data['low'], data['close'].shift(1))
    data['TR'] = np.fmax(data['high'], data['close'].shift(1)) - np.fmin(data['low'], data['close'].shift(1))
    data['Ave_U'] = data['BP'].rolling(7).sum() / data['TR'].rolling(7).sum()
    data['Ave_M'] = data['BP'].rolling(14).sum() / data['TR'].rolling(14).sum()
    data['Ave_D'] = data['BP'].rolling(28).sum() / data['TR'].rolling(28).sum()
    data['UO'] = 100 * ((4 * data['Ave_U']) + (2 * data['Ave_M']) + data['Ave_D']) / 7

    # Stochastic RSI
    data['RSI'] = (data['close'] - data['close'].rolling(14).min()) / (data['close'].rolling(14).max() - data['close'].rolling(14).min())
    data['Stoch_RSI_k'] = data['RSI'].rolling(3).mean()
    data['Stoch_RSI_d'] = data['Stoch_RSI_k'].rolling(3).mean()

    # TRIX
    data['TRIX'] = data['close'].ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean()
    data['TRIX_signal'] = data['TRIX'].ewm(span=9, adjust=False).mean()
    
    # Mass Index
    # data['Mass_Index'] = data['high'] - data['low']
    # data['Mass_Index'] = data['Mass_Index'].rolling(9).sum() / data['Mass_Index'].rolling(9).diff(1).sum()

    # CMO
    data['CMO'] = (data['close'] - data['close'].rolling(9).mean()) / (data['close'].rolling(9).max() - data['close'].rolling(9).min())

    # Chaikin Oscillator
    data['Chaikin_Oscillator'] = data['EMA_3'] - data['EMA_10']

    # MFI
    data['MF'] = ((data['high'] + data['low'] + data['close']) / 3) * data['volume']
    data['MF_up'] = data['MF'].copy()
    data['MF_down'] = data['MF'].copy()
    data['MF_up'][data['close'] > data['close'].shift(1)] = 0
    data['MF_down'][data['close'] < data['close'].shift(1)] = 0
    data['MF_up'] = data['MF_up'].rolling(14).sum()
    data['MF_down'] = data['MF_down'].rolling(14).sum()
    data['MFI'] = 100 - (100 / (1 + (data['MF_up'] / data['MF_down'])))
    data['MFI'] = data['MFI'].fillna(50)

    # OBV
    data['OBV'] = data['volume'].copy()
    data['OBV'][data['close'] > data['close'].shift(1)] = data['volume']
    data['OBV'][data['close'] < data['close'].shift(1)] = -data['volume']
    data['OBV'] = data['OBV'].cumsum()

    return data