import os
try:
    import mesa
    print(f"Mesaのバージョン: {mesa.__version__}")

    # mesaライブラリの親ディレクトリのパスを取得
    mesa_path = os.path.dirname(mesa.__file__)
    print(f"Mesaのインストールパス: {mesa_path}")

    # 'experimental'フォルダのフルパスを作成
    experimental_path = os.path.join(mesa_path, 'experimental')
    print(f"調査対象のパス: {experimental_path}")

    # 'experimental'フォルダが存在するか確認し、中身をリストアップ
    if os.path.exists(experimental_path):
        print("--- `experimental` フォルダは存在します ---")
        print(f"`experimental` フォルダの中身: {os.listdir(experimental_path)}")
    else:
        print("--- `experimental` フォルダが見つかりません ---")

except Exception as e:
    print(f"エラーが発生しました: {e}")