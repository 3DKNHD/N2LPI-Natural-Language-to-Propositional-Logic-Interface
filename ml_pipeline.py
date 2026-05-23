#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid ML pipeline for connective prediction.

Architecture:
  Stage 1 — MultiOutputClassifier (HistGradientBoosting) predicts sub-connectors: ¬, ∧, ∨
  Stage 2 — RandomForest predicts the main connective: →, ↔, ∧, ∨, ¬, ATOMIC

Features: 70+ handcrafted linguistic features + TF-IDF character n-grams (2-4).
Stage 1 predictions are stacked as additional features for Stage 2.
"""

import os
import re
import json
import joblib
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from logic_utils import standardize_formula_variables
from config import NEGATION_START_PHRASES, logger

class HybridPipelineTrainer:
    """Two-stage stacked ML pipeline for Spanish connective prediction.
    
    Trains sub-connector classifiers (¬/∧/∨ presence), then uses their
    predictions as features for a main-connector classifier (→/↔/∧/∨/¬/ATOMIC).
    """
    def __init__(self, data_path: str = 'test_hibrido.json'):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None

        self.clf_multi: Optional[MultiOutputClassifier] = None
        self.clf_main: Optional[RandomForestClassifier] = None

        self._load_data()
        self._extract_features()
        self._define_targets()

    def predict_with_confidence(self, sentence: str) -> Tuple[str, float, Dict[str, bool]]:
        """Predict the main connective with confidence score and sub-connector flags."""
        feat = self.extract_advanced_features(sentence)
        hand_df = pd.DataFrame([feat])
        tfidf_df = self._tfidf_transform_single(sentence)
        X_user = pd.concat([hand_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        X_user = X_user.reindex(columns=self.feature_columns, fill_value=0)

        pred_multi = self.clf_multi.predict(X_user)[0]
        X_user_aug = X_user.copy()
        X_user_aug['pred_neg'] = pred_multi[0]
        X_user_aug['pred_conj'] = pred_multi[1]
        X_user_aug['pred_disj'] = pred_multi[2]

        proba = self.clf_main.predict_proba(X_user_aug)[0]
        confidence = min(proba)
        pred_main = self.clf_main.classes_[proba.argmax()]

        return pred_main, confidence, {
            'has_negation': bool(pred_multi[0]),
            'has_conjunction': bool(pred_multi[1]),
            'has_disjunction': bool(pred_multi[2])
        }

    def _load_data(self) -> None:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info(f"Warning: '{self.data_path}' no encontrado. Usando dataset mínimo.")
            data = [
                {"id": 1, "sentence": "No está lloviendo.", "propositions": {"P": "está lloviendo"}, "formula": "¬P", "connectives": ["¬"]},
                {"id": 2, "sentence": "El sistema no ha fallado.", "propositions": {"P": "el sistema ha fallado"}, "formula": "¬P", "connectives": ["¬"]},
                {"id": 3, "sentence": "Si llueve, me quedo en casa.", "propositions": {"P": "llueve", "Q": "me quedo en casa"}, "formula": "P → Q", "connectives": ["→"]},
                {"id": 4, "sentence": "Si estudias, apruebas el examen.", "propositions": {"P": "estudias", "Q": "apruebas el examen"}, "formula": "P → Q", "connectives": ["→"]},
                {"id": 5, "sentence": "Hace frío y está nublado.", "propositions": {"P": "hace frío", "Q": "está nublado"}, "formula": "P ∧ Q", "connectives": ["∧"]}
            ]
        self.df = pd.DataFrame(data)
        logger.info(f"Dataset cargado: {len(self.df)} oraciones.")

    def extract_advanced_features(self, sentence: str) -> Dict[str, float]:
        """Extract 80+ handcrafted linguistic features from a Spanish sentence.
        
        Groups: conjunction markers, disjunction markers, conditional markers,
        negation markers, biconditional markers, structural features, rare connectives.
        """
        features = {}
        s = sentence.lower().strip()
        len_s = len(s)

        # ── Conjunction markers ──
        features['freq_e_conj'] = len(re.findall(r'\be\b', s))
        features['has_sin_embargo'] = int('sin embargo' in s)
        features['has_no_obstante'] = int('no obstante' in s)
        features['has_a_pesar_de'] = int('a pesar de' in s)
        # ── Disjunction markers ──
        features['freq_u'] = len(re.findall(r'\bu\b', s))
        features['has_u_conj'] = int(bool(re.search(r'\bu\b', s)))
        features['double_o'] = int(bool(re.search(r'\bo\b.+\bo\b', s)))
        features['double_u'] = int(bool(re.search(r'\bu\b.+\bu\b', s)))
        features['has_o_ambas_cosas'] = int('o ambas cosas' in s or 'o ambos' in s)
        # ── Negation markers ──
        features['trailing_no'] = int(bool(re.search(r'\bno\s*$', s)))
        # ── Biconditional markers ──
        features['has_equivalente'] = int('equivalente a' in s and 'equivalente a decir' not in s)
        features['has_equivale'] = int('equivale a' in s)
        features['al_menos_with_u'] = int(bool(re.search(r'(al\s+menos|como\s+m[ií]nimo)\s+\w+\s+u\b', s)))
        features['al_menos_with_o'] = int(bool(re.search(r'(al\s+menos|como\s+m[ií]nimo)\s+\w+\s+o\b', s)))
        # ── Conditional markers ──
        features['freq_si'] = len(re.findall(r'\bsi\b', s))
        features['freq_entonces'] = len(re.findall(r'\bentonces\b', s))
        features['freq_si_y_solo_si'] = len(re.findall(r'si y solo si', s))
        features['freq_y'] = len(re.findall(r'\by\b', s))
        features['freq_o'] = len(re.findall(r'\bo\b', s))
        features['freq_no'] = len(re.findall(r'\bno\b', s))
        features['freq_comma'] = s.count(',')
        features['entonces_after_comma'] = int(re.search(r',\s*entonces', s) is not None)
        features['neg_before_verb'] = int(re.search(r'\bno\s+\w+', s) is not None)

        features['has_necesario_para'] = int(bool(re.search(r'\bnecesari[oa]s?\s+para\b', s)))
        features['has_suficiente_para'] = int(bool(re.search(r'\bsuficiente\s+para\b', s)))
        
        # ── Implication vs negation precedence ──
        strong_implication_markers = [r'\bnecesari[oa]s?\b', r'\bsuficiente\b', r'^si\b', r'\bentonces\b']
        first_impl_idx = len_s
        for pattern in strong_implication_markers:
            match = re.search(pattern, s)
            if match and match.start() < first_impl_idx:
                first_impl_idx = match.start()
                
        first_neg_idx = s.find('no ') if 'no ' in s else len_s
        
        features['implication_dominates_negation'] = int(first_impl_idx < first_neg_idx and first_impl_idx != len_s)

        features['aunque_with_negation'] = int(bool(re.search(r'\baunque\b.*\bno\b', s)))
        
        # ── Sentence-level structural features ──
        features['is_long_complex_sentence'] = int(len_s > 60 and features['freq_comma'] > 0)

        features['wrapper_conditional'] = int(s.startswith('si ') and re.search(r'\bentonces\b', s) is not None)

        features['double_conditional_and'] = int(bool(re.search(r'^si\s+.+?,\s*.+?,\s*y\s+si\s+', s)))

        words = re.findall(r'\b\w+\b', s)
        connectors = {'si', 'entonces', 'y', 'o', 'no', 'solo', 'sí', 'además'}
        content_words = [w for w in words if w not in connectors and len(w) > 2]
        features['num_atomic_hint'] = len(content_words)
        features['implicit_conditional'] = int(re.search(r'si\s+.+,\s*\w+', s) is not None)
        features['implicit_conditional_no_comma'] = int(s.startswith('si ') and ',' not in s)
        features['neg_before_conj'] = int(re.search(r'\bno\s+(y|o)\b', s) is not None)
        features['num_paren'] = s.count('(') + s.count(')')
        features['len_chars'] = len_s
        features['len_words'] = len(words)
        features['comma_ratio'] = features['freq_comma'] / (features['len_words'] + 1e-6)
        features['si_and_entonces'] = int(features['freq_si'] > 0 and features['freq_entonces'] > 0)
        features['neg_not_verb'] = int(re.search(r'\bno\s+(el|la|los|las|un|una)\b', s) is not None)
        features['bicond_phrase'] = int('si y solo si' in s)
        features['y_after_comma'] = len(re.findall(r',\s*y\b', s))
        features['o_after_comma'] = len(re.findall(r',\s*o\b', s))

        features['starts_with_negation_phrase'] = int(any(s.startswith(phrase) for phrase in NEGATION_START_PHRASES))

        negation_tokens = ['no', 'falso', 'es verdad']
        first_si = re.search(r'\bsi\b', s)
        if first_si:
            si_start = first_si.start()
            neg_before_si = any(re.search(r'\b' + re.escape(token) + r'\b', s[:si_start]) for token in negation_tokens)
            features['negation_before_conditional'] = int(neg_before_si)
        else:
            features['negation_before_conditional'] = 0

        first_neg = re.search(r'\b(no|falso)\b', s)
        if first_neg:
            neg_idx = first_neg.start()
            features['negation_distance_to_start'] = neg_idx / len_s if len_s > 0 else 1.0
        else:
            features['negation_distance_to_start'] = 1.0

        strong_markers = [r'\bentonces\b', r'\bo bien\b', r'\bsi y solo si\b']
        first_marker_start = None
        for pattern in strong_markers:
            match = re.search(pattern, s)
            if match:
                start = match.start()
                if first_marker_start is None or start < first_marker_start:
                    first_marker_start = start

        if first_neg and (first_marker_start is None or first_neg.start() < first_marker_start):
            features['negation_dominance'] = 1
        else:
            features['negation_dominance'] = 0

        features['starts_with_si'] = int(s.startswith('si '))
        features['starts_with_no_es'] = int(s.startswith('no es cierto') or s.startswith('es falso'))

        if 'si ' in s and ',' in s:
            antecedente = s.split(',')[0]
            features['disjunction_in_antecedent'] = int(' o ' in antecedente)
            features['conjunction_in_antecedent'] = int(' y ' in antecedente)
        else:
            features['disjunction_in_antecedent'] = 0
            features['conjunction_in_antecedent'] = 0

        features['inverted_si'] = int(' si ' in s and not s.startswith('si ') and 'si y solo si' not in s)
        features['freq_cuando'] = len(re.findall(r'\bcuando\b', s))
        features['has_a_menos_que'] = int('a menos que' in s)
        features['freq_pero'] = len(re.findall(r'\bpero\b', s))

        if ',' in s and ' si ' in s:
            consecuente = s.split(',', 1)[1] if ',' in s else ''
            features['nested_si_in_consequent'] = int(' si ' in consecuente and not consecuente.strip().startswith('si ') and 'si y solo si' not in consecuente)
        else:
            features['nested_si_in_consequent'] = 0

        # ── Rare/additional connectives ──
        features['freq_solo_si'] = int(
            bool(re.search(r'\bsolo\s+si\b', s)) and 'si y solo si' not in s)
        features['freq_siempre_que'] = int(
            bool(re.search(r'\bsiempre\s+(y\s+cuando|que)\b', s)))
        features['freq_salvo_excepto'] = int(
            bool(re.search(r'\b(salvo\s+que|excepto\s+si|a\s+no\s+ser\s+que)\b', s)))
        features['freq_basta'] = int(bool(re.search(r'\bbasta\b', s)))
        features['freq_suficiente'] = int(bool(re.search(r'\bsuficiente\b', s)))
        features['freq_necesario'] = int(bool(re.search(r'\bnecesari[oa]s?\b', s)))
        features['freq_ni'] = len(re.findall(r'\bni\b', s))
        features['ni_ni_pattern'] = int(bool(re.search(r'\bni\b.+\bni\b', s)))
        features['freq_aunque'] = int(bool(re.search(r'\baunque\b', s)))
        features['freq_o_bien'] = int(bool(re.search(r'\bo\s+bien\b', s)))
        features['freq_con_tal_que'] = int(bool(re.search(r'\bcon\s+tal\s+que\b', s)))
        features['freq_en_caso_de'] = int(bool(re.search(r'\ben\s+caso\s+de\b', s)))
        features['has_solo_si_not_bicond'] = int(
            'solo si' in s and 'si y solo si' not in s)

        features['freq_ademas_de'] = int(
            bool(re.search(r'\badem[aá]s\s+de\b', s)))
        features['freq_en_adicion'] = int(
            bool(re.search(r'\ben\s+adici[oó]n\s+a\b', s)))
        features['freq_al_menos'] = int(
            bool(re.search(r'\bal\s+menos\b|\bcomo\s+m[ií]nimo\b', s)))
        features['freq_nec_suf'] = int(
            bool(re.search(r'\bnecesari[oa]s?\s+y\s+suficiente\b', s)))
        features['freq_equivalente_decir'] = int(
            bool(re.search(r'\bequivalente\s+a\s+decir\b', s)))
        features['freq_viceversa'] = int(bool(re.search(r'\bviceversa\b', s)))
        features['freq_no_se_cumple'] = int(
            bool(re.search(r'\bno\s+se\s+cumple\b', s)))

        return features

    def _extract_features(self) -> None:
        hand_df = pd.DataFrame(self.df['sentence'].apply(self.extract_advanced_features).tolist())
        self._tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=300, sublinear_tf=True)
        tfidf_matrix = self._tfidf.fit_transform(self.df['sentence'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

        self.X = pd.concat([hand_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        self.feature_columns = self.X.columns.tolist()
        logger.info(f"Extraídas {hand_df.shape[1]} features artesanales + {tfidf_df.shape[1]} TF-IDF = {self.X.shape[1]} total.")

    def _tfidf_transform_single(self, sentence: str) -> pd.DataFrame:
        vec = self._tfidf.transform([sentence]).toarray()
        return pd.DataFrame(vec, columns=[f'tfidf_{i}' for i in range(vec.shape[1])])

    @staticmethod
    def _get_main_connective(formula: str) -> str:
        stripped = formula.strip()
        depth = 0
        for i, ch in enumerate(stripped):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and ch in {'→', '↔', '∧', '∨'}:
                return ch
        if stripped.startswith('¬'):
            return '¬'
        return 'ATOMIC'

    def _define_targets(self) -> None:
        self.df['main_connective'] = self.df['formula'].apply(self._get_main_connective)
        self.y_main = self.df['main_connective']

        self.df['has_negation'] = self.df['formula'].apply(lambda f: int('¬' in f))
        self.df['has_conjunction'] = self.df['formula'].apply(lambda f: int('∧' in f))
        self.df['has_disjunction'] = self.df['formula'].apply(lambda f: int('∨' in f))
        self.y_multilabel = self.df[['has_negation', 'has_conjunction', 'has_disjunction']]

    def train_sub_connectors(self) -> None:
        """Train the multi-label sub-connector classifiers (¬/∧/∨ presence).
        
        Splits data, fits MultiOutputClassifier, and builds stacked features.
        """
        X_train, X_test, y_multi_train, y_multi_test = train_test_split(
            self.X, self.y_multilabel, test_size=0.2, random_state=42, stratify=self.y_main
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_multi_train = y_multi_train
        self.y_multi_test = y_multi_test

        self.clf_multi = MultiOutputClassifier(
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=6, random_state=42)
        )
        self.clf_multi.fit(X_train, y_multi_train)
        y_multi_pred = self.clf_multi.predict(X_test)

        logger.info("\n--- Rendimiento del clasificador de sub-conectores ---")
        for i, label in enumerate(['has_negation', 'has_conjunction', 'has_disjunction']):
            acc = accuracy_score(y_multi_test.iloc[:, i], y_multi_pred[:, i])
            logger.info(f"{label} accuracy: {acc:.3f}")
        self._create_stacked_features()

    def _create_stacked_features(self) -> None:
        multi_preds_train = self.clf_multi.predict(self.X_train)
        multi_preds_test = self.clf_multi.predict(self.X_test)
        self.X_train_augmented = self.X_train.copy()
        self.X_test_augmented = self.X_test.copy()

        for i, label in enumerate(['pred_neg', 'pred_conj', 'pred_disj']):
            self.X_train_augmented[label] = multi_preds_train[:, i]
            self.X_test_augmented[label] = multi_preds_test[:, i]

    def train_main_connector(self) -> None:
        """Train the main-connector classifier (→/↔/∧/∨/¬/ATOMIC) using augmented features."""
        self.y_main_train = self.y_main[self.X_train.index]
        self.y_main_test = self.y_main[self.X_test.index]

        self.clf_main = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.clf_main.fit(self.X_train_augmented, self.y_main_train)
        y_main_pred = self.clf_main.predict(self.X_test_augmented)

        logger.info("\n--- Rendimiento del clasificador de conector principal (Holdout) ---")
        logger.info(f"Accuracy: {accuracy_score(self.y_main_test, y_main_pred):.3f}")
        logger.info(classification_report(self.y_main_test, y_main_pred, zero_division=0))

    def cross_validate_main(self, n_splits: int = 5) -> None:
        """Run stratified k-fold cross-validation on the main connector classifier."""
        X_full_augmented = self.X.copy()
        full_multi_preds = self.clf_multi.predict(self.X)
        for i, label in enumerate(['pred_neg', 'pred_conj', 'pred_disj']):
            X_full_augmented[label] = full_multi_preds[:, i]

        logger.info("\n--- Validación Cruzada Estratificada ---")
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.clf_main, X_full_augmented, self.y_main, cv=skf, scoring='f1_macro')
            logger.info(f"F1-Macro Promedio ({n_splits}-Fold): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except ValueError as e:
            min_class_size = self.y_main.value_counts().min()
            new_splits = max(2, min(n_splits - 1, min_class_size))
            logger.info(f"No hay suficientes muestras para {n_splits}-Fold (error: {e}). Reintentando con {new_splits} pliegues...")
            if new_splits >= 2:
                skf = StratifiedKFold(n_splits=new_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.clf_main, X_full_augmented, self.y_main, cv=skf, scoring='f1_macro')
                logger.info(f"F1-Macro Promedio ({new_splits}-Fold): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def predict(self, sentence: str) -> Tuple[str, Dict[str, bool]]:
        """Predict main connective and sub-connector flags for a single sentence."""
        feat = self.extract_advanced_features(sentence)
        hand_df = pd.DataFrame([feat])
        
        tfidf_df = self._tfidf_transform_single(sentence)
        
        X_user = pd.concat([hand_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        X_user = X_user.reindex(columns=self.feature_columns, fill_value=0)

        pred_multi = self.clf_multi.predict(X_user)[0]
        
        X_user_aug = X_user.copy()
        X_user_aug['pred_neg'] = pred_multi[0]
        X_user_aug['pred_conj'] = pred_multi[1]
        X_user_aug['pred_disj'] = pred_multi[2]

        pred_main = self.clf_main.predict(X_user_aug)[0]
        
        return pred_main, {
            'has_negation': bool(pred_multi[0]),
            'has_conjunction': bool(pred_multi[1]),
            'has_disjunction': bool(pred_multi[2])
        }

    def save_models(self, filepath: str = 'models.joblib') -> None:
        """Serialize classifiers, TF-IDF vectorizer, and feature columns to disk."""
        try:
            data = {
                'clf_multi': self.clf_multi,
                'clf_main': self.clf_main,
                'tfidf': self._tfidf,
                'feature_columns': self.feature_columns
            }
            joblib.dump(data, filepath)
            logger.info(f"Modelos guardados exitosamente en '{filepath}'.")
        except Exception as e:
            logger.error(f"Error al guardar los modelos: {e}")

    def predict_connective(self, sentence : str) -> Tuple[str,Dict[str,bool]]:
        main_pred, flags = self.predict(sentence)
        return main_pred, flags

    def load_models(self, filepath: str = 'models.joblib') -> bool:
        """Load serialized models from disk. Returns True on success, False if file is missing."""
        if not os.path.exists(filepath):
            return False
        try:
            data = joblib.load(filepath)
            self.clf_multi = data['clf_multi']
            self.clf_main = data['clf_main']
            self._tfidf = data['tfidf']
            self.feature_columns = data['feature_columns']
            logger.info(f"Modelos cargados en memoria desde '{filepath}'.")
            return True
        except Exception as e:
            logger.error(f"Error al cargar los modelos: {e}")
            return False