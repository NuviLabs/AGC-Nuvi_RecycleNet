from .coco import CocoDataset
from .builder import DATASETS
### IMPORT YOUR DATASET FORMAT AND PASS IT IN MYDATASET FUNC

@DATASETS.register_module
class Internet_dataset(CocoDataset):

    CLASSES = ('gwailsaelleodeu', 'gogumamastang', 'soegogibulgogi', 'talkgogiyukgaejang', 'soegogimiyeokguk', 'dwaejigogisuyuk', 'kongnamulmaeunmuchim', 'kkwarigochumyeolchibokkeum', 'gamjaguk', 'gamjachaebokkeum', 'bap', 'gamjatang', 'gyeranguk', 'gogumasaelleodeu', 'gochujapchae', 'gungjungtalkjjim', 'gungjungtteokbokki', 'gimchibokkeum', 'kkoccgetang', 'talkgalbi', 'talkgomtang', 'talkjuk', 'dongaseu', 'dwaejigogigimchibokkeum', 'dwaejibulgogi', 'dubujorim', 'tteokmanduguk', 'tteokbokki', 'mapadubu', 'modeumsosijibokkeum', 'musaengchae', 'miteubolkechapjorim', 'kkakdugi', 'baechudoenjangguk', 'baekgimchi', 'bibimbap', 'saeubokkeumbap', 'saengseonmukbokkeum', 'soegogimuguk', 'sundubujjigae', 'seupageti', 'sigeumchimuchim', 'saengseonmukguk', 'karebap', 'oimuchim', 'ojingeobokkeum', 'ojingeochaebokkeum', 'japchae', 'jeyukbokkeum', 'jogaesalmiyeokguk', 'kongnamulguk', 'keurimtteokbokki', 'tangsuyuk', 'pagimchi', 'kkansyosaeu', 'patalk', 'pusgochudoenjangmuchim', 'heukmibap', 'mandutwigim', 'gamjajorim', 'ueongjapchae', 'gimchijjigae', 'baechugimchi',) ## ADD ALL YOUR CLASSES
