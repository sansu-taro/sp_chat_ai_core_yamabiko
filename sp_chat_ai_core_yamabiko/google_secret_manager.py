"""Google Secret Manager から秘密情報のバージョンを取得します"""

from google.cloud import secretmanager


def get_secret(secret_id: str, project_id: str = "smarthr-customer-support", version_id: str = "latest") -> str:
    """Google Secret Manager から秘密情報のバージョンを取得します。

    :param project_id: GCP プロジェクト ID
    :param secret_id: 秘密情報の ID
    :param version_id: 取得したい秘密情報のバージョン(デフォルトは "latest")
    :return: 秘密情報の値
    """
    # Secret Manager クライアントの作成
    client = secretmanager.SecretManagerServiceClient()

    # 秘密情報のリソース名の組み立て
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # 秘密情報の取得
    response = client.access_secret_version(request={"name": name})

    # 秘密情報の値を文字列として返す
    return response.payload.data.decode("UTF-8")


if __name__ == "__main__":
    pass
