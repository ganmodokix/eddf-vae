from vaetc.models import register_model

from .maskvae import MaskVAE
register_model("maskvae", MaskVAE)
from .vitae2 import VITAE2
register_model("vitae2", VITAE2)
from .asvae import AdaptiveSamplingVAE
register_model("asvae", AdaptiveSamplingVAE)
from .bnvae import BNVAE
register_model("bnvae", BNVAE)
from .hiddenvae import HiddenConsistencyVAE
register_model("hiddenvae", HiddenConsistencyVAE)

from .ivae import MoorePenroseInverseVAE
register_model("ivae", MoorePenroseInverseVAE)
from .ivae2 import MoorePenroseInverseVAE2
register_model("ivae2", MoorePenroseInverseVAE2)
from .ivae4 import MoorePenroseInverseVAE4
register_model("ivae4", MoorePenroseInverseVAE4)
from .ivae8 import MoorePenroseInverseVAE8
register_model("ivae8", MoorePenroseInverseVAE8)

from .adainvae import AdainVAE
register_model("adainvae", AdainVAE)

from .isbvae import IndependentSchedulingBetaVAE
register_model("isbvae", IndependentSchedulingBetaVAE)

from .qivae import QIVAE
register_model("qivae", QIVAE)

from .pwsvae import PixelwiseSigmaVAE
register_model("pwsvae", PixelwiseSigmaVAE)

from .semvae import SelfEncodingMetricVAE
register_model("semvae", SelfEncodingMetricVAE)

from .wuvae import WassersteinUniformVAE
register_model("wuvae", WassersteinUniformVAE)
from .wuinfovae import WassersteinUniformInfoVAE
register_model("wuinfovae", WassersteinUniformInfoVAE)

from .lavae import LaplaceVAE
register_model("lavae", LaplaceVAE)
from .glvae import GlobalLocalVAE
register_model("glvae", GlobalLocalVAE)
from .gnvae import GeneralizedNormalVAE
register_model("gnvae", GeneralizedNormalVAE)
from .emvae import EMVAE
register_model("emvae", EMVAE)

from .gwae import GromovWassersteinAutoEncoder
register_model("gwae", GromovWassersteinAutoEncoder)
from .gwe import GromovWassersteinEncoder
register_model("gwe", GromovWassersteinEncoder)

from .sigmamapvae import SigmaMapVAE
register_model("sigmamapvae", SigmaMapVAE)

from .bnencvae import BNEncoderVAE
register_model("bnencvae", BNEncoderVAE)

from .rechsicvae import ReconstructionHSICVAE
register_model("rechsicvae", ReconstructionHSICVAE)

from .klablatedvae import KLAblatedVAE
register_model("klablatedvae", KLAblatedVAE)

from .essfvae import ExponentialSimilarityScaleFamilyVAE
register_model("essfvae", ExponentialSimilarityScaleFamilyVAE)
from .essfvaegan import ESSFVAEGAN
register_model("essfvaegan", ESSFVAEGAN)
from .essfintrovae import ESSFIntroVAE
register_model("essfintrovae", ESSFIntroVAE)

from .maskedvae import MaskedVAE
register_model("maskedvae", MaskedVAE)

from .lotvae import LatentOptimalTransportVAE
register_model("lotvae", LatentOptimalTransportVAE)

from .maxpoolvae import MaxPoolVAE
register_model("maxpoolvae", MaxPoolVAE)

from .rcvae import RandomConvolutionVAE
register_model("rcvae", RandomConvolutionVAE)

from .vqvae import VQVAE
register_model("vqvae", VQVAE)
from .vunqvae import VUnQVAE
register_model("vunqvae", VUnQVAE)

from .lsgm import LatentScoreBasedGenerativeModel
register_model("lsgm", LatentScoreBasedGenerativeModel)

from .mlpmixervae import MLPMixerVAE
register_model("mlpmixervae", MLPMixerVAE)

from .deepvae import DeepVAE
register_model("deepvae", DeepVAE)


from .sigsigvae import SigSigVAE
register_model("sigsigvae", SigSigVAE)

from .vitvae import ViTVAE
register_model("vitvae", ViTVAE)
