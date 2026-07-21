import React, {useState} from "react";
import {
	View,
	Text,
	Image,
	TouchableOpacity,
	ActivityIndicator,
	StyleSheet,
	SafeAreaView,
	ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import {StatusBar} from "expo-status-bar";

const API_URL = "http://192.168.1.31:8000";

export default function App() {
	const [image, setImage] = useState<string | null>(null);
	const [result, setResult] = useState<{
		class: string;
		confidence: number;
		message?: string;
	} | null>(null);
	const [loading, setLoading] = useState(false);

	const pickImage = async () => {
		const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
		if (!perm.granted) return;
		const res = await ImagePicker.launchImageLibraryAsync({
			mediaTypes: ["images"],
			quality: 1,
		});
		if (!res.canceled) {
			setImage(res.assets[0].uri);
			setResult(null);
		}
	};

	const takePhoto = async () => {
		const perm = await ImagePicker.requestCameraPermissionsAsync();
		if (!perm.granted) return;
		const res = await ImagePicker.launchCameraAsync({
			quality: 1,
		});
		if (!res.canceled) {
			setImage(res.assets[0].uri);
			setResult(null);
		}
	};

	const predict = async () => {
		if (!image) return;
		setLoading(true);
		setResult(null);

		const formData = new FormData();
		formData.append("file", {
			uri: image,
			type: "image/jpeg",
			name: "photo.jpg",
		} as any);

		try {
			const res = await fetch(`${API_URL}/predict`, {
				method: "POST",
				body: formData,
				headers: {"Content-Type": "multipart/form-data"},
			});
			const data = await res.json();
			setResult(data);
		} catch {
			setResult({class: "Error de conexión", confidence: 0});
		} finally {
			setLoading(false);
		}
	};

	return (
		<SafeAreaView style={styles.container}>
			<StatusBar style="light" />
			<ScrollView contentContainerStyle={styles.content}>
				<Text style={styles.title}>Clasificador de Vegetales</Text>

				<View style={styles.buttonRow}>
					<TouchableOpacity style={styles.btn} onPress={takePhoto}>
						<Text style={styles.btnText}>📷 Tomar foto</Text>
					</TouchableOpacity>
					<TouchableOpacity style={styles.btn} onPress={pickImage}>
						<Text style={styles.btnText}>🖼️ Galería</Text>
					</TouchableOpacity>
				</View>

				{image && <Image source={{uri: image}} style={styles.preview} />}

				{image && !loading && (
					<TouchableOpacity style={styles.predictBtn} onPress={predict}>
						<Text style={styles.predictBtnText}>Predecir</Text>
					</TouchableOpacity>
				)}

				{loading && <ActivityIndicator size="large" color="#4CAF50" />}

        {result && (
          <View style={styles.resultBox}>
            <Text style={result.class === "No reconocido" || result.class === "Error de conexión" ? styles.resultClassError : styles.resultClass}>
              {result.class}
            </Text>
            <Text style={styles.resultConf}>
              Confianza: {(result.confidence * 100).toFixed(1)}%
            </Text>
            {result.message && (
              <Text style={styles.resultMsg}>{result.message}</Text>
            )}
          </View>
        )}
			</ScrollView>
		</SafeAreaView>
	);
}

const styles = StyleSheet.create({
	container: {flex: 1, backgroundColor: "#1a1a2e"},
	content: {alignItems: "center", padding: 20},
	title: {
		fontSize: 24,
		fontWeight: "bold",
		color: "#fff",
		marginBottom: 24,
		marginTop: 16,
	},
	buttonRow: {flexDirection: "row", gap: 12, marginBottom: 20},
	btn: {
		backgroundColor: "#16213e",
		paddingVertical: 14,
		paddingHorizontal: 20,
		borderRadius: 12,
	},
	btnText: {color: "#fff", fontSize: 16, fontWeight: "600"},
	preview: {
		width: 280,
		height: 280,
		borderRadius: 16,
		marginBottom: 20,
		borderWidth: 2,
		borderColor: "#4CAF50",
	},
	predictBtn: {
		backgroundColor: "#4CAF50",
		paddingVertical: 16,
		paddingHorizontal: 48,
		borderRadius: 12,
		marginBottom: 20,
	},
	predictBtnText: {color: "#fff", fontSize: 18, fontWeight: "bold"},
	resultBox: {
		backgroundColor: "#16213e",
		borderRadius: 16,
		padding: 24,
		alignItems: "center",
		width: "100%",
	},
	resultClass: {fontSize: 28, fontWeight: "bold", color: "#4CAF50"},
	resultClassError: {fontSize: 28, fontWeight: "bold", color: "#ff6b6b"},
	resultMsg: {fontSize: 14, color: "#ff6b6b", marginTop: 8, textAlign: "center"},
	resultConf: {fontSize: 16, color: "#aaa", marginTop: 8},
});

