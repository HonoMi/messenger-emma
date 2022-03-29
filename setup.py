import setuptools

setuptools.setup(
    name="messenger",
    version="0.1.1",
    author="Austin Wang Hanjie",
    author_email="hjwang@cs.princeton.edu",
    description="Implements EMMA model and Messenger environments.",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'gym',
        'numpy',
        # 'vgdl @ git+https://github.com/ahjwang/py-vgdl',
        'vgdl@git+https://github.com/HonoMi/py-vgdl.git@honoka_dev',
        'pygame',
        'tenacity',
        # 'self-attention-cv',
        'self_attention_cv@git+https://github.com/HonoMi/self-attention-cv.git@honoka-dev',
        'transformers>=4.2',
        'torch>=1.3',
    ],
)
